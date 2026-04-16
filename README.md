# boom-healpix

Benchmark: HEALPix integer range queries vs MongoDB 2dsphere for astronomical cone searches, plus a 2-machine sharding demo.

Standalone — no boom dependency.

## Quick start

```bash
# Start MongoDB instances
docker compose up -d

# Query benchmark (1M catalog)
cargo run --release --bin benchmark -- --catalog-size 1000000 --n-queries 1000

# Sharding demo (2 machines)
cargo run --release --bin sharding-demo -- --n-alerts 100000 --radius-arcsec 10
```

## Benchmark (`benchmark`)

Generates a synthetic catalog, indexes it three ways, and compares:

| Method | How it works |
|--------|-------------|
| **2dsphere** | MongoDB's native `$geoWithin`/`$centerSphere` |
| **HEALPix ranges** | `$or` with `$gte`/`$lt` ranges on depth-29 `hpx` field |
| **HEALPix $in** | `$in` with pixel list at depth 16 (extcats approach) |

All HEALPix results are post-filtered for exact distance.

Reports: query latency, index build time, index size, disk usage, WiredTiger cache utilization, overinclusiveness.

```bash
cargo run --release --bin benchmark -- --help
cargo run --release --bin benchmark -- --catalog-size 1000000 --n-queries 500
cargo run --release --bin benchmark -- --explain  # show MongoDB query plans
```

## Sharding demo (`sharding-demo`)

Two separate MongoDB instances simulate a 2-machine cluster. The sky is split using HEALPix base cells (depth 0, 12 cells):
- Machine 0 (port 27081): base cells 0–5
- Machine 1 (port 27082): base cells 6–11

Demonstrates:
- Alert routing by sky position
- Cone search routing (only relevant machine queried)
- Load balance (~50/50 split)
- Boundary-crossing statistics

```bash
# Typical cross-match radius — 100% single-machine queries
cargo run --release --bin sharding-demo -- --radius-arcsec 10

# Larger cone — shows occasional boundary crossing
cargo run --release --bin sharding-demo -- --radius-arcsec 300

# More alerts
cargo run --release --bin sharding-demo -- --n-alerts 500000 --radius-arcsec 30
```

## CI

GitHub Actions runs both the benchmark and sharding demo on every push/PR. See [`.github/workflows/benchmark.yaml`](.github/workflows/benchmark.yaml).

## Cleanup

```bash
docker compose down -v
```
