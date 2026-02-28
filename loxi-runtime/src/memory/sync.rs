// memory/sync.rs
// ─────────────────────────────────────────────────────────────────────────────
// Shared memory synchronization via Architect.
//
// How shared memory propagates:
//   1. Node marks an entry as MemoryScope::Shared when inserting
//   2. Periodically (every SYNC_INTERVAL), node sends delta to Architect
//   3. Architect broadcasts delta to peers in same affinity group
//   4. Peers call merge_shared() to absorb new entries
//
// This is eventually-consistent: entries may arrive out of order.
// Conflict resolution: same UUID = same entry (UUIDs are globally unique).
//
// Wire protocol reuses the existing QUIC transport from orchestrator.
// MemorySyncMessage is encoded as bincode and sent as a LoxiMessage payload.
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::time;

use super::store::{MemoryEntry, MemoryStore};

// ─── Sync interval ────────────────────────────────────────────────────────────

/// How often to push shared memory delta to Architect.
const SYNC_INTERVAL: Duration = Duration::from_secs(30);

/// Max entries per sync message (prevents large payloads).
const MAX_ENTRIES_PER_SYNC: usize = 100;

// ─── Wire Messages ────────────────────────────────────────────────────────────

/// Sent from node → Architect when new shared entries are available.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemorySyncMessage {
    /// The node that's publishing these entries
    pub source_node: String,
    /// Affinity domain of the entries (for targeted gossip)
    pub domain:      String,
    /// New entries to broadcast (lz4-compressed bincode inside)
    pub entries:     Vec<MemoryEntry>,
    /// Unix timestamp of this sync
    pub timestamp:   u64,
}

/// Sent from Architect → node with merged entries from peers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemorySyncResponse {
    /// Entries from peers to merge
    pub peer_entries: Vec<MemoryEntry>,
    /// How many nodes contributed
    pub peer_count:   u32,
}

// ─── Sync State ───────────────────────────────────────────────────────────────

/// Tracks which entries have already been synced to avoid re-sending.
struct SyncState {
    last_synced_count: usize,
}

// ─── Sync Client ──────────────────────────────────────────────────────────────

/// Runs in the background, periodically syncing shared memory.
pub struct MemorySyncClient {
    node_id:    String,
    domain:     String,
    store:      Arc<MemoryStore>,
    state:      Mutex<SyncState>,
    // Sender for outgoing sync messages (connected to QUIC transport)
    // Type-erased to avoid circular dep with transport layer.
    // In practice: tokio::sync::mpsc::Sender<MemorySyncMessage>
    tx:         tokio::sync::mpsc::Sender<MemorySyncMessage>,
    rx:         Mutex<tokio::sync::mpsc::Receiver<MemorySyncResponse>>,
}

impl MemorySyncClient {
    /// Create a sync client.
    ///
    /// tx: channel to send sync messages to the transport layer
    /// rx: channel to receive merged entries from the transport layer
    pub fn new(
        node_id: String,
        domain:  String,
        store:   Arc<MemoryStore>,
        tx:      tokio::sync::mpsc::Sender<MemorySyncMessage>,
        rx:      tokio::sync::mpsc::Receiver<MemorySyncResponse>,
    ) -> Self {
        Self {
            node_id,
            domain,
            store,
            state: Mutex::new(SyncState { last_synced_count: 0 }),
            tx,
            rx: Mutex::new(rx),
        }
    }

    /// Start the background sync loop.
    /// Spawns a tokio task — call once at node startup.
    pub fn start(self: Arc<Self>) {
        let client = Arc::clone(&self);
        tokio::spawn(async move {
            client.sync_loop().await;
        });
    }

    async fn sync_loop(&self) {
        let mut interval = time::interval(SYNC_INTERVAL);
        loop {
            interval.tick().await;
            if let Err(e) = self.push_delta().await {
                tracing::warn!("Memory sync push failed: {}", e);
            }
            if let Err(e) = self.pull_updates().await {
                tracing::warn!("Memory sync pull failed: {}", e);
            }
        }
    }

    /// Push new shared entries to Architect.
    async fn push_delta(&self) -> Result<()> {
        let all_entries = self.store.export_shared().await;

        let mut state = self.state.lock().await;
        let new_entries: Vec<MemoryEntry> = all_entries
            .into_iter()
            .skip(state.last_synced_count)
            .take(MAX_ENTRIES_PER_SYNC)
            .collect();

        if new_entries.is_empty() {
            return Ok(());
        }

        let count = new_entries.len();
        let msg = MemorySyncMessage {
            source_node: self.node_id.clone(),
            domain:      self.domain.clone(),
            entries:     new_entries,
            timestamp:   unix_now(),
        };

        self.tx.send(msg).await
            .map_err(|_| anyhow::anyhow!("Memory sync tx channel closed"))?;

        state.last_synced_count += count;
        tracing::debug!("Pushed {} shared memory entries to network", count);

        Ok(())
    }

    /// Pull merged entries from peers (non-blocking, drains the channel).
    async fn pull_updates(&self) -> Result<()> {
        let mut rx = self.rx.lock().await;

        let mut total_merged = 0;
        while let Ok(response) = rx.try_recv() {
            let merged = self.store.merge_shared(response.peer_entries).await;
            total_merged += merged;
        }

        if total_merged > 0 {
            tracing::info!("Merged {} shared memory entries from peers", total_merged);
        }

        Ok(())
    }

    /// Immediately push a high-priority memory (bypass interval).
    /// Used when a node discovers something important mid-session.
    pub async fn push_now(&self, entries: Vec<MemoryEntry>) -> Result<()> {
        if entries.is_empty() { return Ok(()); }

        let msg = MemorySyncMessage {
            source_node: self.node_id.clone(),
            domain:      self.domain.clone(),
            entries,
            timestamp:   unix_now(),
        };

        self.tx.send(msg).await
            .map_err(|_| anyhow::anyhow!("Memory sync tx channel closed"))?;

        Ok(())
    }
}

// ─── Architect-side handler (runs in the orchestrator) ────────────────────────
//
// When Architect receives a MemorySyncMessage:
//   1. Store in affinity-group memory pool
//   2. Fan out to all other nodes in same affinity group
//
// This is implemented in the orchestrator (scheduler.rs), not here.
// This file just defines the message types and the node-side client.

fn unix_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
