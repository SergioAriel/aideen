use aideen_training::data::DataLoader;
use std::io::Write;

/// Helper: build a simple token corpus of sequential u32 values.
fn sample_tokens(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

#[test]
fn dataloader_produces_correct_batch_size() {
    let ctx_len = 8;
    let tokens = sample_tokens(64);
    let mut dl = DataLoader::new(tokens, ctx_len, 42);

    let (input, target) = dl.next_batch();
    assert_eq!(input.len(), ctx_len, "input length must equal ctx_len");
    assert_eq!(target.len(), ctx_len, "target length must equal ctx_len");
}

#[test]
fn dataloader_targets_are_shifted_inputs() {
    let ctx_len = 16;
    let tokens = sample_tokens(128);
    let mut dl = DataLoader::new(tokens, ctx_len, 7);

    let (input, target) = dl.next_batch();

    // target[i] == input[i+1] for all i in 0..ctx_len-1
    for i in 0..ctx_len - 1 {
        assert_eq!(
            target[i],
            input[i + 1],
            "target[{}] should equal input[{}]",
            i,
            i + 1
        );
    }

    // Additionally, since our corpus is sequential integers, verify the
    // overall shift: target is input shifted right by 1.
    // target[0] == input[1] (already checked) and target[ctx_len-1] == input[0] + ctx_len
    // which is the token one past the end of input.
    assert_eq!(
        target[ctx_len - 1],
        input[0] + ctx_len as u32,
        "last target element should be one past the end of input window"
    );
}

#[test]
fn dataloader_wraps_around() {
    // Use a small corpus and pull 100 batches - should never panic.
    let ctx_len = 4;
    let tokens = sample_tokens(10); // 10 tokens, ctx_len=4 => window of 5, max_start=5
    let mut dl = DataLoader::new(tokens.clone(), ctx_len, 123);

    for batch_idx in 0..100 {
        let (input, target) = dl.next_batch();
        assert_eq!(input.len(), ctx_len, "batch {} input length", batch_idx);
        assert_eq!(target.len(), ctx_len, "batch {} target length", batch_idx);

        // All values must be within the token range.
        for &tok in input.iter().chain(target.iter()) {
            assert!(
                (tok as usize) < tokens.len(),
                "token {} out of range at batch {}",
                tok,
                batch_idx
            );
        }
    }
}

#[test]
fn dataloader_from_file_loads_binary() {
    let ctx_len = 4;
    let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];

    // Write tokens as little-endian u32 to a temp file.
    let dir = std::env::temp_dir().join("aideen_data_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokens.bin");

    {
        let mut f = std::fs::File::create(&path).unwrap();
        for &t in &tokens {
            f.write_all(&t.to_le_bytes()).unwrap();
        }
    }

    let mut dl = DataLoader::from_file(&path, ctx_len, 99).expect("from_file should succeed");

    assert_eq!(dl.len(), tokens.len(), "loaded token count must match");

    let (input, target) = dl.next_batch();
    assert_eq!(input.len(), ctx_len);
    assert_eq!(target.len(), ctx_len);

    // Verify every token in the batch actually exists in our original data.
    for &tok in input.iter().chain(target.iter()) {
        assert!(tokens.contains(&tok), "token {} not in original data", tok);
    }

    // Clean up.
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
