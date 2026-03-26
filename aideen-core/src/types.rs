/// Result of a system tick
#[derive(Debug, Clone)]
pub enum TickResult {
    Integrated,
    Rejected,
    NoOp,
}
