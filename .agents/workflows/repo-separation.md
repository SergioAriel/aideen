---
description: Procedure for repo separation and maintenance of private logic.
---

# Workflow: Repository Separation & Privacy

Procedure to ensure the `aideen-training` logic remains private while maintaining the public `aideen` workspace.

1. **Feature Gating**: Use Rust features or workspace members to isolate `autograd` and `optimizer`.
2. **Code Mirroring**:
   - Updates to `aideen-core` and `aideen-runtime` are pushed to the public repo.
   - `aideen-training` logic remains in the `/Users/sergiosolis/Programacion/aideen-training-lab` directory.
3. **Safety Check**: Before every release, run `cargo test` in a clean environment to ensure no private symbols are exposed in the public crates.
