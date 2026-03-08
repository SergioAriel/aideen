/// Modos de control
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlMode {
    /// V0: Observar y medir sin intervenir (Modo Transparente)
    Observe,
    /// V1: Regular, penalizar y estabilizar
    Regulate,
}

/// Decisión de control por iteración
#[derive(Debug, Clone)]
pub struct ControlDecision {
    /// ¿Debe frenarse el bucle DEQ?
    pub stop: bool,
    /// Coeficiente de intensidad de integración (Atenuación)
    pub beta: f32,
    /// ¿Se autoriza la persistencia en memoria local?
    pub write_memory: bool,
    /// ¿Se autoriza el aprendizaje local (ajuste de pesos)?
    pub allow_learning: bool,
}

/// Control de convergencia y parada (El orquestador)
pub trait Control {
    /// Límite máximo de iteraciones permitidas
    fn max_iters(&self) -> usize;

    /// Modo actual de operación
    fn mode(&self) -> ControlMode;

    /// Tomar una decisión basada en la trayectoria y entropía actual
    fn decide(&self, iter: usize, delta_norm: f32, entropy: f32) -> ControlDecision;

    /// Función opcional para registrar métricas de trayectoria (V0)
    fn observe(&self, _iter: usize, _delta_norm: f32, _entropy: f32) {}
}
