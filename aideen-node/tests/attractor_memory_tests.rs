use aideen_core::{memory::Memory, state::ArchitectureConfig};
use aideen_node::memory::{AttractorMemory, NullMemory};
use nalgebra::DVector;

// ── Helpers ───────────────────────────────────────────────────────────────

fn vec2(x: f32, y: f32) -> DVector<f32> {
    DVector::from_vec(vec![x, y])
}

fn zeros(dim: usize) -> DVector<f32> {
    DVector::zeros(dim)
}

fn test_config() -> ArchitectureConfig {
    ArchitectureConfig::default()
}

// ── Test 1 ────────────────────────────────────────────────────────────────

/// query on empty index → vec![]
#[test]
fn test_query_empty_before_inserts() {
    let mem = AttractorMemory::new(2);
    let result = mem.query(&vec2(1.0, 0.0), 5);
    assert!(result.is_empty(), "índice vacío debe devolver vec![]");
}

// ── Test 2 ────────────────────────────────────────────────────────────────

/// Vec A similar to the query; Vec B orthogonal → A ranks first by cosine.
#[test]
fn test_cosine_ordering_similar_before_orthogonal() {
    let mut mem = AttractorMemory::new(2);

    // A = very similar to (1,0)
    let a = vec2(0.99, 0.01);
    // B = orthogonal to (1,0)
    let b = vec2(0.0, 1.0);

    mem.write(b.clone());
    mem.write(a.clone());

    let results = mem.query(&vec2(1.0, 0.0), 2);
    assert_eq!(results.len(), 2);

    // The first result must be a (most similar to (1,0))
    let first = &results[0];
    let cos_first = first.dot(&vec2(1.0, 0.0)) / first.norm();
    let cos_second = results[1].dot(&vec2(1.0, 0.0)) / results[1].norm();
    assert!(
        cos_first >= cos_second,
        "first result must have higher cosine similarity than the second"
    );
}

// ── Test 3 ────────────────────────────────────────────────────────────────

/// k > n → returns n results (not k).
#[test]
fn test_insert_k_clamps_to_n() {
    let mut mem = AttractorMemory::new(2);
    mem.write(vec2(1.0, 0.0));
    mem.write(vec2(0.0, 1.0));
    mem.write(vec2(1.0, 1.0));

    let results = mem.query(&vec2(1.0, 0.0), 10);
    assert_eq!(results.len(), 3, "k=10 con 3 elementos debe devolver 3");
}

// ── Test 4 ────────────────────────────────────────────────────────────────

/// Wrong dimension in write → panic with clear message.
#[test]
#[should_panic(expected = "dim mismatch en write")]
fn test_dim_mismatch_panics_on_write() {
    let mut mem = AttractorMemory::new(4);
    mem.write(DVector::zeros(2)); // dim=2, mem espera dim=4
}

// ── Test 5 ────────────────────────────────────────────────────────────────

/// NullMemory + AideenNode-style warm-start: query returns vec![], fallback without panic.
/// Validates that the DEQ loop warm-start pattern is safe with NullMemory.
#[test]
fn test_warm_start_falls_back_with_null_memory() {
    let mut mem = NullMemory;
    let config = test_config();
    let d_res = config.d_reasoning();
    let query = zeros(d_res);

    let results = mem.query(&query, 1);
    assert!(results.is_empty(), "NullMemory.query debe devolver vec![]");

    // Simular warm-start del DEQ: si empty, usa init() (simulado como zeros)
    let h_init = zeros(d_res);
    let h = results.into_iter().next().unwrap_or(h_init.clone());
    assert_eq!(h, h_init, "warm-start con NullMemory debe producir h_init");
}
