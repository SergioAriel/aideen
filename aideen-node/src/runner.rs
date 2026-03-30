#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;

use nalgebra::DVector;
use sha2::{Digest, Sha256};

use aideen_core::{
    agent::{AgentEvent, AgentStore},
    capabilities::NodeCapabilities,
    compute::ComputeBackend,
    control::Control,
    doc_memory::{DocHit, DocId, DocMemory, DocMeta},
    ethics::Ethics,
    memory::Memory,
    protocol::NetMsg,
    reasoning::Reasoning,
};

use std::sync::Arc;

use crate::network::channel_factory::{ChannelFactory, NullChannelFactory};
use crate::peers::{NodeId, PeerDelta, PeerEntry, PeerFailures, PeerRegistry};
use crate::security::trust_store::TrustStore;
use crate::system::node::{AideenNode, StopReason, TickMetrics};

// ── ReconcileStats ────────────────────────────────────────────────────────────

/// Diagnostic counters for the last reconcile cycle.
/// Local metrics only — not sent over the network.
#[derive(Default, Debug, Clone)]
pub struct ReconcileStats {
    pub dial_attempts: u32,
    pub dial_success: u32,
    pub dial_fail: u32,
    pub trust_ok: u32,
    pub trust_fail: u32,
    pub breaker_skips: u32,
}

// ── RouterStatsAccumulator ────────────────────────────────────────────────────

/// Accumulates routing metrics per window of N ticks.
/// The caller calls `flush()` periodically and sends the result over QUIC
/// (same pattern as `discovery` in `tick_with_query`).
pub struct RouterStatsAccumulator {
    flush_every: u32,
    ticks: u32,
    q_samples: Vec<f32>,
    expert_hits: std::collections::HashMap<String, u32>,
    // Stability Pack:
    delta_norms: Vec<f32>,
    drops_total: u32,
    beta_samples: Vec<f32>,
}

impl RouterStatsAccumulator {
    pub fn new(flush_every: u32) -> Self {
        Self {
            flush_every,
            ticks: 0,
            q_samples: Vec::new(),
            expert_hits: Default::default(),
            delta_norms: Vec::new(),
            drops_total: 0,
            beta_samples: Vec::new(),
        }
    }

    /// Records a quality sample and the queried expert (optional).
    pub fn record(&mut self, q: f32, target_id: Option<&str>) {
        self.q_samples.push(q);
        self.ticks += 1;
        if let Some(id) = target_id {
            *self.expert_hits.entry(id.to_string()).or_insert(0) += 1;
        }
    }

    /// Records statistics from a query to the expert pipeline.
    pub fn record_expert(&mut self, delta_norm: f32, drops: u32, beta: f32) {
        self.delta_norms.push(delta_norm);
        self.drops_total += drops;
        self.beta_samples.push(beta);
    }

    /// Returns `NetMsg::RouterStats` if `flush_every` ticks have been reached, and resets.
    /// Returns `None` if not enough data has been accumulated yet.
    pub fn flush(&mut self, node_id: [u8; 32]) -> Option<NetMsg> {
        if self.ticks < self.flush_every || self.q_samples.is_empty() {
            return None;
        }
        let n = self.q_samples.len() as f32;
        let q_mean = self.q_samples.iter().sum::<f32>() / n;
        let q_min = self.q_samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let q_max = self
            .q_samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut expert_hits: Vec<(String, u32)> = self.expert_hits.drain().collect();
        expert_hits.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic order for tests
        let window_ticks = self.ticks;
        self.ticks = 0;
        self.q_samples.clear();

        let delta_norm_mean = mean_or_zero(&self.delta_norms);
        let delta_norm_min = min_or_zero(&self.delta_norms);
        let delta_norm_max = max_or_zero(&self.delta_norms);
        let beta_mean = mean_or_zero(&self.beta_samples);
        let drops_count = self.drops_total;
        self.delta_norms.clear();
        self.beta_samples.clear();
        self.drops_total = 0;

        Some(NetMsg::RouterStats {
            node_id,
            window_ticks,
            q_mean,
            q_min,
            q_max,
            expert_hits,
            unix_ts: unix_now(),
            delta_norm_mean,
            delta_norm_min,
            delta_norm_max,
            drops_count,
            beta_mean,
        })
    }
}

// ── Tipos públicos ────────────────────────────────────────────────────────────

/// Result of a tick with document context injection.
///
/// `discovery` is only present when `metrics.allow_learning == true`
/// AND the runner has `delegated == true` (Zero-Trust gate).
/// The caller is responsible for sending `discovery` over the QUIC channel.
pub struct TickOutcome {
    pub metrics: TickMetrics,
    pub discovery: Option<NetMsg>,
}

// ── NodeRunner ────────────────────────────────────────────────────────────────

/// High-level entry point for an AIDEEN node.
///
/// Wraps `AideenNode<R,C,E,M,B>` and adds:
/// - Automatic emission of `AgentEvent::TickAttractor` when h* is reached.
/// - Document memory API (add/search/locate).
/// - Document context injection into S_sim before tick.
/// - Construction of `NetMsg::Discovery` when `allow_learning && delegated`.
///
/// `node_id`, `bundle_version` and `target_id` are the node's stable identity
/// and are fixed at construction. `set_delegated()` is updated by the caller
/// when `UpdateManager` installs a valid `KeyDelegation`.
pub struct NodeRunner<R, C, E, M, B> {
    pub node: AideenNode<R, C, E, M, B>,
    pub agent_store: Box<dyn AgentStore + Send>,
    pub doc_memory: Box<dyn DocMemory + Send>,
    pub node_id: [u8; 32],
    pub bundle_version: u64,
    pub target_id: String,
    /// Device capabilities detected at construction.
    pub caps: NodeCapabilities,
    /// Routing metrics accumulator. Call `flush()` every N ticks.
    pub stats_acc: RouterStatsAccumulator,
    /// Indexed peer directory (PeerRegistry with by_id + by_domain).
    pub peer_registry: PeerRegistry,
    // β damping params (Stability Pack):
    /// Base damping factor. β = clamp(β0 * q_mean / (1 + delta_norm), β_min, β_max).
    pub beta0: f32,
    pub beta_min: f32,
    pub beta_max: f32,
    // 5L — dial + security:
    /// QUIC channel factory. Default: NullChannelFactory (no real dial).
    pub channel_factory: Arc<dyn ChannelFactory>,
    /// TOFU + TLS fingerprint pinning per peer.
    pub trust_store: TrustStore,
    /// Circuit breakers per NodeId — exponential backoff on dial failures.
    pub peer_failures: PeerFailures,
    // 5M — telemetría operativa:
    /// Counters from the last reconcile cycle. Local diagnostics only.
    pub last_reconcile_stats: ReconcileStats,
    delegated: bool, // gate Zero-Trust: Delegation instalada
}

impl<R, C, E, M, B> NodeRunner<R, C, E, M, B>
where
    R: Reasoning,
    C: Control,
    E: Ethics,
    M: Memory,
    B: ComputeBackend,
{
    /// Canonical constructor. `delegated` starts at `false`; use
    /// `set_delegated(true)` upon receiving a valid `KeyDelegation`.
    pub fn new(
        node: AideenNode<R, C, E, M, B>,
        agent_store: Box<dyn AgentStore + Send>,
        doc_memory: Box<dyn DocMemory + Send>,
        node_id: [u8; 32],
        bundle_version: u64,
        target_id: String,
    ) -> Self {
        Self {
            node,
            agent_store,
            doc_memory,
            node_id,
            bundle_version,
            target_id,
            caps: crate::capabilities::detect(),
            stats_acc: RouterStatsAccumulator::new(100),
            peer_registry: PeerRegistry::new(),
            beta0: 1.0,
            beta_min: 0.05,
            beta_max: 1.0,
            channel_factory: Arc::new(NullChannelFactory),
            trust_store: TrustStore::new(),
            peer_failures: PeerFailures::default(),
            last_reconcile_stats: ReconcileStats::default(),
            delegated: false,
        }
    }

    /// Enables/disables the Zero-Trust Discovery gate.
    /// Call with `true` when `UpdateManager` installs a valid `KeyDelegation`.
    pub fn set_delegated(&mut self, v: bool) {
        self.delegated = v;
    }

    // ── PeerRegistry API ──────────────────────────────────────────────────────

    /// Full directory bootstrap: replaces all state with `peers`.
    pub fn set_peer_snapshot(&mut self, epoch: u64, peers: Vec<PeerEntry>) {
        self.peer_registry.set_snapshot(epoch, peers);
    }

    /// Applies an incremental delta. Rejects if `delta.epoch <= current epoch`.
    pub fn apply_peer_delta(&mut self, delta: PeerDelta) -> Result<(), String> {
        self.peer_registry.apply_delta(delta)
    }

    /// NodeIds of peers serving `domain`, deterministic order (sorted).
    pub fn peer_ids_for_domain(&self, domain: &str) -> Vec<NodeId> {
        self.peer_registry.node_ids_for_domain(domain)
    }

    // ── 5L: dial + reconcile ──────────────────────────────────────────────────

    /// Synchronises `client` with `PeerRegistry` for the given `domain`.
    ///
    /// - Removes stale peers from the client (retain_only).
    /// - Attempts to dial new peers, respecting circuit breakers.
    /// - Applies TOFU/pinning in TrustStore before adding the channel.
    /// - Updates `last_reconcile_stats` with cycle counters.
    pub fn reconcile_expert_client(
        &mut self,
        domain: &str,
        client: &mut crate::expert::ExpertClient,
    ) {
        use std::collections::HashSet;

        let mut stats = ReconcileStats::default();

        let domain_ids = self.peer_registry.node_ids_for_domain(domain);
        let domain_set: HashSet<NodeId> = domain_ids.iter().copied().collect();
        client.retain_only(&domain_set);

        for id in &domain_ids {
            if client.has_peer(id) {
                continue;
            }

            let breaker = self.peer_failures.entry(*id).or_default();
            if !breaker.can_try() {
                stats.breaker_skips += 1;
                continue;
            }

            let entry = match self.peer_registry.get(id) {
                Some(e) => e.clone(),
                None => continue,
            };

            stats.dial_attempts += 1;
            match self.channel_factory.dial(&entry) {
                Ok(dr) => {
                    match self.trust_store.verify_or_tofu(
                        *id,
                        dr.fingerprint,
                        entry.tls_fingerprint,
                    ) {
                        Ok(_) => {
                            stats.dial_success += 1;
                            stats.trust_ok += 1;
                            breaker.record_success();
                            client.upsert_peer(*id, dr.channel);
                        }
                        Err(e) => {
                            stats.trust_fail += 1;
                            stats.dial_fail += 1;
                            breaker.record_failure(id);
                            eprintln!("[runner] trust error for peer: {e}");
                        }
                    }
                }
                Err(e) => {
                    stats.dial_fail += 1;
                    breaker.record_failure(id);
                    eprintln!("[runner] dial error for peer: {e}");
                }
            }
        }

        self.last_reconcile_stats = stats;
    }

    // ── Expert result injection ───────────────────────────────────────────────

    /// Applies damping β to the pipeline RunResult and injects into the node's R space.
    /// β = clamp(β0 * q_mean / (1.0 + delta_norm), β_min, β_max).
    /// No intermediate allocation (inject_delta_r_scaled). Returns the applied β.
    pub fn apply_expert_result(&mut self, result: &crate::expert::RunResult) -> f32 {
        let raw = self.beta0 * result.q_mean / (1.0 + result.delta_norm);
        let beta = raw.clamp(self.beta_min, self.beta_max);
        self.node.inject_delta_r_scaled(&result.delta, beta);
        self.stats_acc
            .record_expert(result.delta_norm, result.drops_count, beta);
        beta
    }

    // ── Tick base ─────────────────────────────────────────────────────────────

    /// Runs a cognitive tick without context injection.
    /// If an attractor is reached, emits `AgentEvent::TickAttractor`.
    pub fn tick(&mut self) -> Option<TickMetrics> {
        let metrics = self.node.tick()?;

        self.stats_acc
            .record(metrics.quality.q_total, Some(&self.target_id));

        if metrics.is_attractor {
            let h_star_hash: [u8; 32] = if let Some(ref h) = metrics.h_star {
                let flat = h.to_flat();
                let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
                Sha256::digest(&bytes).into()
            } else {
                [0u8; 32]
            };

            let _ = self.agent_store.append_event(AgentEvent::TickAttractor {
                q_total: metrics.quality.q_total,
                iters: metrics.iters as u32,
                stop: stop_reason_to_u8(metrics.stop_reason),
                h_star_hash,
                unix_ts: unix_now(),
            });
        }

        Some(metrics)
    }

    // ── DocMemory API ─────────────────────────────────────────────────────────

    /// Inserts a document into the document memory.
    pub fn add_document(&mut self, meta: DocMeta, bytes: Vec<u8>) -> Result<DocId, String> {
        self.doc_memory.add_document(meta, bytes)
    }

    /// Lexical search in the document memory. Returns top-k hits.
    pub fn search_docs(&self, query: &str, k: usize) -> Vec<DocHit> {
        self.doc_memory.search(query, k)
    }

    /// Offsets byte-exactos de `needle` en el documento raw.
    pub fn locate(&self, doc_id: DocId, needle: &[u8], limit: usize) -> Vec<(u64, u64)> {
        self.doc_memory.locate(doc_id, needle, limit)
    }

    // ── AgentStore API ────────────────────────────────────────────────────────

    /// Last `limit` agent events (reverse chronological order).
    pub fn recent_events(&self, limit: usize) -> Vec<AgentEvent> {
        self.agent_store.recent_events(limit)
    }

    // ── 6A: Replay handler ────────────────────────────────────────────────────

    /// Handles a `NetMsg::ReplayRequest` from the Coordinator.
    ///
    /// Light replay for 6A: recomputes a deterministic `trace_digest` from the request
    /// fields combined with this node's identity. Full DEQ re-execution is 6N scope.
    /// `q_recomputed` is a deterministic stub derived from the seed.
    pub fn handle_replay_request(&self, req: &NetMsg) -> NetMsg {
        match req {
            NetMsg::ReplayRequest {
                sample_id,
                context_hash,
                iters,
                seed,
                ..
            } => {
                // trace: sha256(node_id ‖ sample_id ‖ context_hash ‖ iters)
                let mut buf = Vec::with_capacity(76);
                buf.extend_from_slice(&self.node_id);
                buf.extend_from_slice(&sample_id.to_le_bytes());
                buf.extend_from_slice(context_hash);
                buf.extend_from_slice(&iters.to_le_bytes());
                let trace_digest: [u8; 32] = Sha256::digest(&buf).into();

                // Deterministic stub: seed mod 1000, normalised to [0, 1)
                let q_recomputed = (seed % 1000) as f32 / 1000.0;

                NetMsg::ReplayResponse {
                    sample_id: *sample_id,
                    reproduced: true,
                    q_recomputed,
                    trace_digest,
                }
            }
            _ => NetMsg::Error {
                code: 400,
                msg: "expected ReplayRequest".into(),
            },
        }
    }

    // ── Tick con contexto + Discovery ─────────────────────────────────────────

    /// Tick with document context injection.
    ///
    /// 1. Searches `query` in `doc_memory` → features → `node.set_context()`
    /// 2. Calls `tick()` (emits `TickAttractor` if applicable)
    /// 3. If `metrics.allow_learning && delegated`:
    ///    - Builds `NetMsg::Discovery` (hashes, no raw h*)
    ///    - Emits `AgentEvent::DiscoveryEmitted` in agent_store
    ///    - Returns `discovery` in `TickOutcome`
    ///
    /// The caller decides when/if to send `discovery` over the QUIC channel.
    pub fn tick_with_query(&mut self, query: &str, k: usize) -> Option<TickOutcome> {
        // 1. Context injection
        let hits = self.doc_memory.search(query, k);
        let events = self.agent_store.recent_events(16);

        let ctx = RuntimeContext {
            docs: hits,
            prefs: HashMap::new(),
            recent_events: events,
        };
        let d_sim = self.node.reasoning.config().d_sim;
        let features = build_context_features(&ctx, d_sim);
        self.node.set_context(&features);

        // 2. Tick base (emite TickAttractor si aplica)
        let metrics = self.tick()?;

        // 3. Discovery gate: allow_learning && delegated
        let discovery = if metrics.allow_learning && self.delegated {
            if let Some(ref h_star) = metrics.h_star {
                // Hashes: solo viajan hashes, nunca h* raw (Zero-Trust)
                let h_flat = h_star.to_flat();
                let h_bytes: Vec<u8> = h_flat.iter().flat_map(|f| f.to_le_bytes()).collect();
                let h_star_hash: [u8; 32] = Sha256::digest(&h_bytes).into();

                let ctx_bytes: Vec<u8> = self
                    .node
                    .state
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let context_hash: [u8; 32] = Sha256::digest(&ctx_bytes).into();

                let msg = NetMsg::Discovery {
                    node_id: self.node_id,
                    target_id: self.target_id.clone(),
                    q_total: metrics.quality.q_total,
                    iters: metrics.iters as u32,
                    stop: stop_reason_to_u8(metrics.stop_reason),
                    h_star_hash,
                    context_hash,
                    bundle_version: self.bundle_version,
                };

                let _ = self.agent_store.append_event(AgentEvent::DiscoveryEmitted {
                    q_total: metrics.quality.q_total,
                    iters: metrics.iters as u32,
                    target_id: self.target_id.clone(),
                    bundle_version: self.bundle_version,
                    unix_ts: unix_now(),
                });

                Some(msg)
            } else {
                None
            }
        } else {
            None
        };

        Some(TickOutcome { metrics, discovery })
    }
}

// ── Helpers privados ──────────────────────────────────────────────────────────

fn mean_or_zero(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn min_or_zero(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().cloned().fold(f32::INFINITY, f32::min)
    }
}

fn max_or_zero(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
}

fn stop_reason_to_u8(r: StopReason) -> u8 {
    match r {
        StopReason::Control => 0,
        StopReason::Epsilon => 1,
        StopReason::Ethics => 2,
        StopReason::LowQuality => 3,
        StopReason::ReachedAttractor => 4,
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Context injection ─────────────────────────────────────────────────────────

/// Runtime context aggregated before each tick.
///
/// Packages document search hits, agent preferences and
/// recent events. `build_context_features()` projects it into a
/// vector of dimension D_SIM for injection into S_sim.
pub struct RuntimeContext {
    pub docs: Vec<DocHit>,
    pub prefs: HashMap<String, String>,
    pub recent_events: Vec<AgentEvent>,
}

/// Projects a `RuntimeContext` into a feature vector of dimension `dim`.
///
/// Algorithm: deterministic slot-hashing for each source:
/// - DocHits: hash(doc_id, chunk_id) → slot, accumulates score
/// - Prefs:   FNV-1a(key) → slot, accumulates 1.0
/// - Events:  type discriminant → slot, accumulates 1.0
/// Final normalisation: per-component tanh → range (-1, 1].
pub fn build_context_features(ctx: &RuntimeContext, dim: usize) -> DVector<f32> {
    let mut feats = vec![0.0f32; dim];

    // DocHits: multiplicative hash (doc_id, chunk_id) → slot
    for hit in &ctx.docs {
        let h = hit
            .doc_id
            .wrapping_mul(2654435761)
            .wrapping_add((hit.chunk_id as u64).wrapping_mul(40503));
        feats[(h as usize) % dim] += hit.score;
    }

    // Prefs: FNV-1a hash of key string → slot
    for key in ctx.prefs.keys() {
        feats[fnv1a(key.as_bytes()) % dim] += 1.0;
    }

    // Events: fixed type discriminant → slot
    for ev in &ctx.recent_events {
        feats[event_slot(ev, dim)] += 1.0;
    }

    // Normalize via tanh → (-1, 1]
    DVector::from_vec(feats.into_iter().map(|x| x.tanh()).collect())
}

fn fnv1a(bytes: &[u8]) -> usize {
    let mut h: u64 = 14695981039346656037u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h as usize
}

fn event_slot(ev: &AgentEvent, dim: usize) -> usize {
    let disc: usize = match ev {
        AgentEvent::TickAttractor { .. } => 0,
        AgentEvent::UpdateApplied { .. } => 1,
        AgentEvent::DelegationInstalled { .. } => 2,
        AgentEvent::DiscoveryEmitted { .. } => 3,
        AgentEvent::PreferenceSet { .. } => 4,
    };
    (disc * 97 + 17) % dim
}
