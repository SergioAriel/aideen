FROM rust:1.85-slim AS builder

WORKDIR /app
COPY . .

# Build workspace (CPU-only, no GPU features)
RUN cargo build --release --workspace --exclude aideen-block --exclude aideen-engine

# Run tests
RUN cargo test --workspace \
    --exclude aideen-block \
    --exclude aideen-engine \
    --exclude aideen-backbone \
    --exclude aideen-node \
    --exclude aideen-training

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/arch_bench /usr/local/bin/aideen-bench
ENTRYPOINT ["aideen-bench"]
