use std::collections::HashMap;

use aideen_core::artifacts::ArtifactId;

pub struct ArtifactStore {
    data: HashMap<ArtifactId, Vec<u8>>,
}

impl ArtifactStore {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: ArtifactId, bytes: Vec<u8>) {
        self.data.insert(id, bytes);
    }

    pub fn get(&self, id: &ArtifactId) -> Option<&[u8]> {
        self.data.get(id).map(|v| v.as_slice())
    }

    pub fn list(&self) -> Vec<&ArtifactId> {
        self.data.keys().collect()
    }
}
