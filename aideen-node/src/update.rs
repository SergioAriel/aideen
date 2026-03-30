use aideen_backbone::ffn_reasoning::{FfnReasoning, ReplayGuard};
use aideen_core::agent::{AgentEvent, AgentStore, NullAgentStore};
use aideen_core::protocol::{KeyDelegation, ModelBundleManifest, SignedUpdate};

// En native: Send requerido por tokio::spawn / QUIC.
// En WASM:   single-threaded, Send no aplica (OpfsAgentStore puede usar Rc internamente).
#[cfg(not(target_arch = "wasm32"))]
type AgentStoreBox = Box<dyn AgentStore + Send>;
#[cfg(target_arch = "wasm32")]
type AgentStoreBox = Box<dyn AgentStore>;

pub struct UpdateManager {
    pub root_pk: [u8; 32],
    pub current_critic_pk: Option<[u8; 32]>,
    pub current_epoch: u64,
    pub bundle_version: u64,
    pub bundle_hash: [u8; 32],
    pub guard: ReplayGuard,
    agent_store: AgentStoreBox,
}

impl UpdateManager {
    pub fn new(root_pk: [u8; 32]) -> Self {
        Self {
            root_pk,
            current_critic_pk: None,
            current_epoch: 0,
            bundle_version: 0,
            bundle_hash: [0u8; 32],
            guard: ReplayGuard::default(),
            agent_store: Box::new(NullAgentStore),
        }
    }

    /// Builder: inyectar un backend de agente concreto (FsAgentStore, InMemory, etc.).
    pub fn with_agent_store(mut self, store: AgentStoreBox) -> Self {
        self.agent_store = store;
        self
    }

    pub fn apply_delegation(&mut self, d: &KeyDelegation) -> Result<(), String> {
        d.verify_signature(&self.root_pk)?;

        if d.epoch <= self.current_epoch {
            return Err("KeyDelegation rollback detected (epoch <= current)".into());
        }

        self.current_epoch = d.epoch;
        self.current_critic_pk = Some(d.critic_pk);

        // Emitir evento de telemetría (ignora error del store)
        let critic_pk_hash = {
            use sha2::{Digest, Sha256};
            Sha256::digest(&d.critic_pk).into()
        };
        let _ = self
            .agent_store
            .append_event(AgentEvent::DelegationInstalled {
                epoch: d.epoch,
                critic_pk_hash,
                unix_ts: unix_now(),
            });

        Ok(())
    }

    pub fn critic_pk(&self) -> Result<[u8; 32], String> {
        self.current_critic_pk
            .ok_or("No critic_pk installed (missing KeyDelegation)".into())
    }

    pub fn apply_manifest(&mut self, manifest_bytes: &[u8]) -> Result<(), String> {
        let m: ModelBundleManifest =
            bincode::deserialize(manifest_bytes).map_err(|e| e.to_string())?;
        let bytes = bincode::serialize(&m).map_err(|e| e.to_string())?;
        use sha2::{Digest, Sha256};
        let h = Sha256::digest(&bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(&h);

        self.bundle_version = m.bundle_version;
        self.bundle_hash = out;
        Ok(())
    }

    pub fn apply_update_bytes(
        &mut self,
        expert: &mut FfnReasoning,
        update_bytes: &[u8],
    ) -> Result<(), String> {
        let update: SignedUpdate = bincode::deserialize(update_bytes).map_err(|e| e.to_string())?;
        self.apply_update_struct(expert, &update)
    }

    pub fn apply_update_struct(
        &mut self,
        expert: &mut FfnReasoning,
        update: &SignedUpdate,
    ) -> Result<(), String> {
        // 1) Bundle gate
        if update.bundle_version < self.bundle_version {
            return Err("update bundle_version older than local".into());
        }
        if update.bundle_hash != self.bundle_hash {
            return Err("bundle_hash mismatch".into());
        }

        // 2) Requerir Critic PK válida
        let pk = self.critic_pk()?;

        // 3) Aplica en backbone
        expert.apply_signed_update(update, &pk, &mut self.guard)?;

        // Emitir evento de telemetría.
        // update_hash() is based on signing_bytes() (see aideen-core/src/protocol.rs:91-96).
        let update_hash = update.update_hash();
        let _ = self.agent_store.append_event(AgentEvent::UpdateApplied {
            version: update.version,
            target_id: update.target_id.clone(),
            update_hash,
            unix_ts: unix_now(),
        });

        Ok(())
    }

    /// Enrutador principal de nivel 6 (Transporte Escalable). Consumido por clientes WASM o Nativos al recibir un NetMsg del Coordinator.
    pub fn on_message(
        &mut self,
        expert: &mut FfnReasoning,
        msg: aideen_core::protocol::NetMsg,
    ) -> Result<aideen_core::protocol::NetMsg, String> {
        use aideen_core::protocol::{AckKind, NetMsg};

        match msg {
            NetMsg::Delegation(d) => {
                self.apply_delegation(&d)?;
                Ok(NetMsg::Ack {
                    kind: AckKind::Delegation,
                    version: d.epoch,
                    ok: true,
                })
            }
            NetMsg::Update(u) => {
                let v = u.version;
                match self.apply_update_struct(expert, &u) {
                    Ok(_) => Ok(NetMsg::Ack {
                        kind: AckKind::Update,
                        version: v,
                        ok: true,
                    }),
                    Err(e) => Err(e),
                }
            }
            NetMsg::Ping => Ok(NetMsg::Pong),
            _ => Err("Mensaje no procesable por UpdateManager en este contexto".into()),
        }
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
