#![cfg(not(target_arch = "wasm32"))]

use aideen_core::doc_memory::DocMemory;
use aideen_node::doc_memory::FsDocMemory;

// ── Helper ────────────────────────────────────────────────────────────────────

fn tmp_dir() -> std::path::PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("aideen_docmem_test_{}", n))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn meta(title: &str) -> aideen_core::doc_memory::DocMeta {
    aideen_core::doc_memory::DocMeta {
        title: title.into(),
        locator: format!("file:{}.txt", title),
        mime: "text/plain".into(),
        len_bytes: 0,
        added_unix: unix_now(),
    }
}

// ── Test 1 ────────────────────────────────────────────────────────────────────

/// add_document + search → hit found with correct doc_id and preview containing the token.
#[test]
fn add_doc_then_search_finds_hit() {
    let base = tmp_dir();
    let mut mem = FsDocMemory::open(base.to_str().unwrap(), "agentA").unwrap();

    let bytes = b"hello world\nthis is a large book\nworld of aideen\n".to_vec();
    let doc_id = mem.add_document(meta("Book1"), bytes).unwrap();

    let hits = mem.search("world", 5);
    assert!(!hits.is_empty(), "search must find at least one hit");
    assert_eq!(hits[0].doc_id, doc_id);
    assert!(
        hits[0].preview.to_lowercase().contains("world"),
        "preview must contain the searched token"
    );
}

// ── Test 2 ────────────────────────────────────────────────────────────────────

/// locate returns byte-exact offsets of two occurrences of the needle.
#[test]
fn locate_returns_exact_offsets() {
    let base = tmp_dir();
    let mut mem = FsDocMemory::open(base.to_str().unwrap(), "agentB").unwrap();

    // "abc mundo abc mundo abc"
    //  0123456789...
    //  "mundo" starts at 4 (exclusive: 9) and at 14 (exclusive: 19)
    let bytes = b"abc mundo abc mundo abc".to_vec();
    let doc_id = mem.add_document(meta("t"), bytes).unwrap();

    let offs = mem.locate(doc_id, b"mundo", 10);
    assert!(offs.len() >= 2, "must find the 2 occurrences");
    assert_eq!(offs[0], (4, 9), "first occurrence at bytes [4,9)");
    assert_eq!(offs[1], (14, 19), "second occurrence at bytes [14,19)");
}

// ── Test 3 ────────────────────────────────────────────────────────────────────

/// get_chunk returns the chunk containing the searched token.
#[test]
fn get_chunk_returns_matching_context() {
    let base = tmp_dir();
    let mut mem = FsDocMemory::open(base.to_str().unwrap(), "agentC").unwrap();

    let bytes = b"one two three four five six seven eight nine ten".to_vec();
    let doc_id = mem.add_document(meta("nums"), bytes).unwrap();

    let hits = mem.search("eight", 1);
    assert_eq!(hits.len(), 1);

    let chunk = mem.get_chunk(doc_id, hits[0].chunk_id).unwrap();
    let text = String::from_utf8_lossy(&chunk).to_lowercase();
    assert!(
        text.contains("eight"),
        "chunk must contain the token 'eight'"
    );
}

// ── Test 4 ────────────────────────────────────────────────────────────────────

/// open → add → drop → open → search: the data persists after restart.
#[test]
fn survives_restart() {
    let base = tmp_dir();

    {
        let mut mem = FsDocMemory::open(base.to_str().unwrap(), "agentD").unwrap();
        let bytes = b"persistencia aideen doc memory sprint 5D".to_vec();
        let _ = mem.add_document(meta("persist"), bytes).unwrap();
        let hits = mem.search("persistencia", 5);
        assert!(!hits.is_empty(), "hits before the restart");
    }

    // Reopen — simulates a process restart
    let mem2 = FsDocMemory::open(base.to_str().unwrap(), "agentD").unwrap();
    let hits2 = mem2.search("persistencia", 5);
    assert!(!hits2.is_empty(), "hits must survive the restart");
}
