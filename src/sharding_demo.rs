//! Sharding demo: two MongoDB instances simulating a 2-machine cluster.
//!
//! Splits the sky using HEALPix base cells (depth 0, 12 cells):
//!   Machine 0 (port 27081): base cells 0–5
//!   Machine 1 (port 27082): base cells 6–11
//!
//! Demonstrates:
//!   1. Alert routing — each alert goes to the correct machine based on sky position
//!   2. Cone search routing — only the relevant machine(s) are queried
//!   3. Load balance — alerts distribute roughly evenly
//!   4. Post-filter — exact distance check on HEALPix candidates
//!
//! Usage:
//!   docker compose up mongo-shard-0 mongo-shard-1 -d
//!   cargo run --release --bin sharding-demo

use cdshealpix::nested::{self, get};
use clap::Parser;
use comfy_table::{Cell, Table};
use futures::stream::StreamExt;
use mongodb::bson::doc;
use mongodb::options::ClientOptions;
use rand::Rng;
use std::time::Instant;

const HPX_DEPTH: u8 = 29;
const QUERY_DEPTH: u8 = 16;
const N_BASE_CELLS: i64 = 12;

#[derive(Parser)]
#[command(name = "sharding-demo", about = "Two-machine HEALPix sharding demo")]
struct Args {
    /// MongoDB URI for machine 0
    #[arg(long, default_value = "mongodb://localhost:27081")]
    shard0_uri: String,

    /// MongoDB URI for machine 1
    #[arg(long, default_value = "mongodb://localhost:27082")]
    shard1_uri: String,

    /// Number of alerts to scatter across the sky
    #[arg(long, default_value_t = 100_000)]
    n_alerts: usize,

    /// Number of cone searches to run
    #[arg(long, default_value_t = 200)]
    n_queries: usize,

    /// Cone search radius in arcseconds
    #[arg(long, default_value_t = 10.0)]
    radius_arcsec: f64,
}

fn hpx_value(ra_deg: f64, dec_deg: f64, depth: u8) -> i64 {
    get(depth).hash(ra_deg.to_radians(), dec_deg.to_radians()) as i64
}

/// Route to machine: base cells 0–5 → machine 0, 6–11 → machine 1.
fn route(hpx29: i64) -> usize {
    let base_cell = hpx29 >> (2 * HPX_DEPTH as i64);
    if base_cell < N_BASE_CELLS / 2 { 0 } else { 1 }
}

/// Build HEALPix range filter on the depth-29 `hpx` field.
/// Uses $or with $gte/$lt ranges — works at any cone size.
fn hpx_range_filter(ra: f64, dec: f64, radius_rad: f64) -> (mongodb::bson::Document, usize) {
    let bmoc = nested::cone_coverage_approx(
        QUERY_DEPTH,
        ra.to_radians(),
        dec.to_radians(),
        radius_rad,
    );
    let ranges = bmoc.to_ranges();
    let shift = 2 * (HPX_DEPTH - QUERY_DEPTH);
    let n = ranges.len();

    let filter = if n == 1 {
        doc! {
            "hpx": {
                "$gte": (ranges[0].start << shift) as i64,
                "$lt": (ranges[0].end << shift) as i64,
            }
        }
    } else {
        let clauses: Vec<mongodb::bson::Document> = ranges.iter().map(|r| {
            doc! {
                "hpx": {
                    "$gte": (r.start << shift) as i64,
                    "$lt": (r.end << shift) as i64,
                }
            }
        }).collect();
        doc! { "$or": clauses }
    };
    (filter, n)
}

fn angular_distance_arcsec(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (ra1, dec1) = (ra1.to_radians(), dec1.to_radians());
    let (ra2, dec2) = (ra2.to_radians(), dec2.to_radians());
    let dlat = dec2 - dec1;
    let dlon = ra2 - ra1;
    let a = (dlat / 2.0).sin().powi(2) + dec1.cos() * dec2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().asin().to_degrees() * 3600.0
}

fn fmt_bytes(bytes: i64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

async fn connect(uri: &str, name: &str) -> Result<mongodb::Database, Box<dyn std::error::Error>> {
    let mut opts = ClientOptions::parse(uri).await?;
    opts.server_selection_timeout = Some(std::time::Duration::from_secs(10));
    let client = mongodb::Client::with_options(opts)?;
    let db = client.database("boom_shard_demo");
    db.run_command(doc! { "ping": 1 }).await
        .map_err(|e| format!("{} at {}: {}", name, uri, e))?;
    Ok(db)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let radius_rad = (args.radius_arcsec / 3600.0_f64).to_radians();

    // --- Connect to both machines ---
    println!("Connecting...");
    let db0 = connect(&args.shard0_uri, "Machine 0").await?;
    let db1 = connect(&args.shard1_uri, "Machine 1").await?;
    let machines = [&db0, &db1];
    println!("  Machine 0: {} (base cells 0–5)", args.shard0_uri);
    println!("  Machine 1: {} (base cells 6–11)", args.shard1_uri);

    let coll_name = "alerts";
    for m in &machines {
        let _ = m.collection::<mongodb::bson::Document>(coll_name).drop().await;
    }
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Create hpx index on both machines
    for m in &machines {
        let coll = m.collection::<mongodb::bson::Document>(coll_name);
        coll.create_index(
            mongodb::IndexModel::builder().keys(doc! { "hpx": 1 }).build(),
        ).await?;
    }

    // --- Insert alerts ---
    println!("\n=== Inserting {} alerts ===\n", args.n_alerts);
    let mut rng = rand::rng();
    let mut counts = [0usize; 2];
    let batch_size = 5000;
    let mut batches: [Vec<mongodb::bson::Document>; 2] = [Vec::new(), Vec::new()];

    let t = Instant::now();
    for i in 0..args.n_alerts {
        let ra: f64 = rng.random::<f64>() * 360.0;
        let u: f64 = rng.random::<f64>();
        let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
        let h29 = hpx_value(ra, dec, HPX_DEPTH);
        let machine_idx = route(h29);

        batches[machine_idx].push(doc! {
            "_id": i as i64,
            "ra": ra,
            "dec": dec,
            "hpx": h29,
        });
        counts[machine_idx] += 1;

        // Flush batches
        for m_idx in 0..2 {
            if batches[m_idx].len() >= batch_size {
                let batch = std::mem::take(&mut batches[m_idx]);
                machines[m_idx]
                    .collection::<mongodb::bson::Document>(coll_name)
                    .insert_many(batch)
                    .await?;
            }
        }
    }
    // Flush remaining
    for m_idx in 0..2 {
        if !batches[m_idx].is_empty() {
            let batch = std::mem::take(&mut batches[m_idx]);
            machines[m_idx]
                .collection::<mongodb::bson::Document>(coll_name)
                .insert_many(batch)
                .await?;
        }
    }
    let insert_time = t.elapsed();

    println!("  Machine 0: {} alerts ({:.1}%)", counts[0], counts[0] as f64 / args.n_alerts as f64 * 100.0);
    println!("  Machine 1: {} alerts ({:.1}%)", counts[1], counts[1] as f64 / args.n_alerts as f64 * 100.0);
    println!("  Insert time: {:.2}s", insert_time.as_secs_f64());

    // --- Disk/index sizes per machine ---
    println!("\nStorage per machine:");
    let mut size_tbl = Table::new();
    size_tbl.set_header(vec!["", "Machine 0", "Machine 1"]);

    let mut data_sizes = [0i64; 2];
    let mut idx_sizes = [0i64; 2];
    for (i, m) in machines.iter().enumerate() {
        let coll = m.collection::<mongodb::bson::Document>(coll_name);
        let stats: Vec<_> = coll.aggregate(vec![doc! { "$collStats": { "storageStats": {} } }])
            .await?.collect::<Vec<_>>().await;
        if let Some(Ok(doc)) = stats.first() {
            if let Ok(s) = doc.get_document("storageStats") {
                let get = |k: &str| -> i64 {
                    match s.get(k) {
                        Some(mongodb::bson::Bson::Int64(v)) => *v,
                        Some(mongodb::bson::Bson::Int32(v)) => *v as i64,
                        Some(mongodb::bson::Bson::Double(v)) => *v as i64,
                        _ => 0,
                    }
                };
                data_sizes[i] = get("storageSize");
                idx_sizes[i] = get("totalIndexSize");
            }
        }
    }
    size_tbl.add_row(vec![
        Cell::new("Data on disk"),
        Cell::new(fmt_bytes(data_sizes[0])),
        Cell::new(fmt_bytes(data_sizes[1])),
    ]);
    size_tbl.add_row(vec![
        Cell::new("Indexes"),
        Cell::new(fmt_bytes(idx_sizes[0])),
        Cell::new(fmt_bytes(idx_sizes[1])),
    ]);
    size_tbl.add_row(vec![
        Cell::new("Total"),
        Cell::new(fmt_bytes(data_sizes[0] + idx_sizes[0])),
        Cell::new(fmt_bytes(data_sizes[1] + idx_sizes[1])),
    ]);
    println!("{}", size_tbl);

    // --- Cone search demo ---
    println!("\n=== Cone search: {} queries at {}\" radius ===\n", args.n_queries, args.radius_arcsec);

    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(args.n_queries);
    for _ in 0..args.n_queries {
        let ra: f64 = rng.random::<f64>() * 360.0;
        let u: f64 = rng.random::<f64>();
        let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
        positions.push((ra, dec));
    }

    // Determine which machine(s) each cone needs
    let mut queries_to_machine = [0usize; 2];
    let mut queries_to_both = 0usize;
    let mut total_exact_hits = 0usize;
    let mut total_raw_hits = 0usize;
    let mut total_machines_queried = 0usize;

    let t = Instant::now();
    for &(ra, dec) in &positions {
        // Which base cells does this cone overlap?
        let shard_bmoc = nested::cone_coverage_approx(
            0, // depth 0 = 12 base cells
            ra.to_radians(),
            dec.to_radians(),
            radius_rad,
        );
        let target_cells: Vec<i64> = shard_bmoc.to_ranges().iter()
            .flat_map(|r| (r.start..r.end).map(|x| x as i64))
            .collect();

        let mut need = [false; 2];
        for cell in &target_cells {
            if *cell < N_BASE_CELLS / 2 { need[0] = true; } else { need[1] = true; }
        }

        if need[0] && need[1] { queries_to_both += 1; }
        if need[0] { queries_to_machine[0] += 1; }
        if need[1] { queries_to_machine[1] += 1; }

        // Query needed machines
        let (filter, _) = hpx_range_filter(ra, dec, radius_rad);
        for (idx, needed) in need.iter().enumerate() {
            if !needed { continue; }
            total_machines_queried += 1;
            let coll = machines[idx].collection::<mongodb::bson::Document>(coll_name);
            let mut cursor = coll.find(filter.clone()).await?;
            while let Some(r) = cursor.next().await {
                let doc = r?;
                total_raw_hits += 1;
                let src_ra = doc.get_f64("ra").unwrap_or(0.0);
                let src_dec = doc.get_f64("dec").unwrap_or(0.0);
                if angular_distance_arcsec(ra, dec, src_ra, src_dec) <= args.radius_arcsec {
                    total_exact_hits += 1;
                }
            }
        }
    }
    let query_time = t.elapsed();

    let avg_machines = total_machines_queried as f64 / args.n_queries as f64;
    let pct_single = (args.n_queries - queries_to_both) as f64 / args.n_queries as f64 * 100.0;

    println!("Query routing:");
    println!("  Queries to machine 0 only: {}", queries_to_machine[0] - queries_to_both);
    println!("  Queries to machine 1 only: {}", queries_to_machine[1] - queries_to_both);
    println!("  Queries to BOTH machines:  {}", queries_to_both);
    println!("  Single-machine queries:    {:.0}%", pct_single);
    println!("  Avg machines per query:    {:.2}", avg_machines);
    println!();
    println!("Results:");
    println!("  Raw hits (pre-filter):  {}", total_raw_hits);
    println!("  Exact hits (post-filter): {}", total_exact_hits);
    println!("  Query time: {:.2}s ({:.1} queries/sec)",
        query_time.as_secs_f64(),
        args.n_queries as f64 / query_time.as_secs_f64()
    );
    println!();

    // Highlight the key point
    println!("=== Key result ===");
    println!("  At {}\" radius, {:.0}% of cone searches hit only 1 of 2 machines.",
        args.radius_arcsec, pct_single);
    println!("  The other machine is never contacted — zero network traffic, zero load.");
    if queries_to_both > 0 {
        println!("  {} queries ({:.1}%) crossed a base-cell boundary and needed both machines.",
            queries_to_both, queries_to_both as f64 / args.n_queries as f64 * 100.0);
    }

    // Cleanup
    for m in &machines {
        let _ = m.collection::<mongodb::bson::Document>(coll_name).drop().await;
    }
    println!("\nCollections dropped.");

    Ok(())
}
