/// Resultado de un tick del sistema
#[derive(Debug, Clone)]
pub enum TickResult {
    Integrated,
    Rejected,
    NoOp,
}
