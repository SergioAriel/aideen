use std::collections::{BTreeMap, HashSet};

use aideen_core::protocol::NetMsg;

use crate::network::NetChannel;
use crate::peers::types::NodeId;

pub struct ExpertClient {
    pub peers: BTreeMap<NodeId, Box<dyn NetChannel>>,
}

impl ExpertClient {
    /// Builds from a list of (node_id, channel) pairs.
    /// BTreeMap guarantees natural order by NodeId — deterministic without extra sort.
    pub fn new(peers: Vec<(NodeId, Box<dyn NetChannel>)>) -> Self {
        Self {
            peers: peers.into_iter().collect(),
        }
    }

    /// Hot-swaps all channels.
    pub fn replace_peers(&mut self, peers: Vec<(NodeId, Box<dyn NetChannel>)>) {
        self.peers = peers.into_iter().collect();
    }

    /// NodeIds in deterministic order (BTreeMap natural).
    pub fn sorted_peer_ids(&self) -> Vec<NodeId> {
        self.peers.keys().copied().collect()
    }

    /// Returns true if the NodeId already has an active channel.
    pub fn has_peer(&self, id: &NodeId) -> bool {
        self.peers.contains_key(id)
    }

    /// Inserts or replaces channel for a NodeId.
    pub fn upsert_peer(&mut self, id: NodeId, ch: Box<dyn NetChannel>) {
        self.peers.insert(id, ch);
    }

    /// Removes peers whose NodeId is NOT in `keep`. O(n) — HashSet lookup O(1).
    pub fn retain_only(&mut self, keep: &HashSet<NodeId>) {
        self.peers.retain(|id, _| keep.contains(id));
    }

    /// Sends `task` to selected peers and collects results.
    /// `selected`: `(node_id, alpha)` already translated from indices in `ExpertPipeline`.
    /// MVP: sequential. Production: `std::thread::scope` for real parallelism (1 RTT).
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
