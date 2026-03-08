use std::collections::{BTreeMap, HashSet};

use aideen_core::protocol::NetMsg;

use crate::network::NetChannel;
use crate::peers::types::NodeId;

pub struct ExpertClient {
    pub peers: BTreeMap<NodeId, Box<dyn NetChannel>>,
}

impl ExpertClient {
    /// Construye desde lista de pares (node_id, canal).
    /// BTreeMap garantiza orden natural por NodeId — determinista sin sort extra.
    pub fn new(peers: Vec<(NodeId, Box<dyn NetChannel>)>) -> Self {
        Self {
            peers: peers.into_iter().collect(),
        }
    }

    /// Reemplaza todos los canales en caliente.
    pub fn replace_peers(&mut self, peers: Vec<(NodeId, Box<dyn NetChannel>)>) {
        self.peers = peers.into_iter().collect();
    }

    /// NodeIds en orden determinista (BTreeMap natural).
    pub fn sorted_peer_ids(&self) -> Vec<NodeId> {
        self.peers.keys().copied().collect()
    }

    /// Devuelve true si el NodeId ya tiene canal activo.
    pub fn has_peer(&self, id: &NodeId) -> bool {
        self.peers.contains_key(id)
    }

    /// Inserta o reemplaza canal para un NodeId.
    pub fn upsert_peer(&mut self, id: NodeId, ch: Box<dyn NetChannel>) {
        self.peers.insert(id, ch);
    }

    /// Elimina peers cuyo NodeId NO está en `keep`. O(n) — HashSet lookup O(1).
    pub fn retain_only(&mut self, keep: &HashSet<NodeId>) {
        self.peers.retain(|id, _| keep.contains(id));
    }

    /// Envía `task` a los peers seleccionados y recoge resultados.
    /// `selected`: `(node_id, alpha)` ya traducidos de índices en `ExpertPipeline`.
    /// MVP: secuencial. Producción: `std::thread::scope` para paralelismo real (1 RTT).
    pub fn query(
        &mut self,
        task: NetMsg,
        selected: &[(NodeId, f32)],
    ) -> Vec<(f32, Result<NetMsg, String>)> {
        selected
            .iter()
            .map(|(node_id, alpha)| {
                let r = (|| {
                    let ch = self
                        .peers
                        .get_mut(node_id)
                        .ok_or_else(|| "peer not found".to_string())?;
                    ch.send(task.clone())?;
                    ch.recv()
                })();
                (*alpha, r)
            })
            .collect()
    }
}
