use crate::state::ArchitectureConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};

/// Unified container for AIDEEN models (.aidn).
/// Packages the architecture configuration and binary weights into a single file.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AidenModel {
    /// Technical configuration of the model.
    pub config: ArchitectureConfig,
    /// Binary weights map (name -> vector of floats).
    pub weights: HashMap<String, Vec<f32>>,
    /// Additional metadata (author, date, version, etc).
    pub metadata: HashMap<String, String>,
}

impl AidenModel {
    pub fn new(config: ArchitectureConfig) -> Self {
        Self {
            config,
            weights: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Saves the model to disk using bincode.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let encoded: Vec<u8> =
            bincode::serialize(self).map_err(|e| format!("Serialization error: {}", e))?;
        file.write_all(&encoded)
            .map_err(|e| format!("Failed to write to file: {}", e))?;
        Ok(())
    }

    /// Loads the model from disk.
    pub fn load(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        let decoded: Self =
            bincode::deserialize(&buffer).map_err(|e| format!("Deserialization error: {}", e))?;
        Ok(decoded)
    }

    /// Inserts a weight into the map.
    pub fn set_weight(&mut self, name: &str, data: Vec<f32>) {
        self.weights.insert(name.to_string(), data);
    }

    /// Gets a weight from the map.
    pub fn get_weight(&self, name: &str) -> Option<&Vec<f32>> {
        self.weights.get(name)
    }
}
