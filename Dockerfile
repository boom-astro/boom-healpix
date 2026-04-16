FROM rust:latest AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo build --release --bin sharded-cluster --bin benchmark --bin sharding-demo

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/sharded-cluster /app/target/release/benchmark /app/target/release/sharding-demo ./
ENTRYPOINT ["./sharded-cluster"]
