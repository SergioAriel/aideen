---
description: Procedure for repo separation and maintenance of private logic.
---

# Workflow: Repository Separation & Privacy

Procedure to ensure the `loxi-training` logic remains private while maintaining the public `loxi-ai` workspace.

1. **Feature Gating**: Use Rust features or workspace members to isolate `autograd` and `optimizer`.
2. **Code Mirroring**:
   - Updates to `loxi-core` and `loxi-runtime` are pushed to the public repo.
   - `loxi-training` logic remains in the `/Users/sergiosolis/Programacion/loxi-training-lab` directory.
3. **Safety Check**: Before every release, run `cargo test` in a clean environment to ensure no private symbols are exposed in the public crates.
