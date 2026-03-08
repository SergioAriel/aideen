use crate::state::ArchitectureConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};

/// Contenedor unificado para modelos AIDEEN (.aidn).
/// Empaqueta la configuración de la arquitectura y los pesos binarios en un solo archivo.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AidenModel {
    /// Configuración técnica del modelo.
    pub config: ArchitectureConfig,
    /// Mapa de pesos binarios (nombre -> vector de floats).
    pub weights: HashMap<String, Vec<f32>>,
    /// Metadatos adicionales (autor, fecha, versión, etc).
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

    /// Guarda el modelo en disco usando bincode.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let encoded: Vec<u8> =
            bincode::serialize(self).map_err(|e| format!("Serialization error: {}", e))?;
        file.write_all(&encoded)
            .map_err(|e| format!("Failed to write to file: {}", e))?;
        Ok(())
    }

    /// Carga el modelo desde disco.
    pub fn load(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        let decoded: Self =
            bincode::deserialize(&buffer).map_err(|e| format!("Deserialization error: {}", e))?;
        Ok(decoded)
    }

    /// Inserta un peso en el mapa.
    pub fn set_weight(&mut self, name: &str, data: Vec<f32>) {
        self.weights.insert(name.to_string(), data);
    }

    /// Obtiene un peso del mapa.
    pub fn get_weight(&self, name: &str) -> Option<&Vec<f32>> {
        self.weights.get(name)
    }
}
