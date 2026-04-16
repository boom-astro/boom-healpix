//! Sharding demo: two independent MongoDB instances with margin overlap.
//!
//! Each machine owns a sky region (6 of 12 HEALPix base cells) and holds:
//!   - All sources in its region (primary data)
//!   - A margin buffer: copies of sources from the OTHER machine that are
//!     within `margin_arcsec` of the region boundary (LSDB-inspired)
//!
//! With margins, every cone search is answered by a SINGLE machine —
//! no cross-shard queries needed, ever. This is application-level
//! partitioning, not MongoDB sharding.
//!
//! Usage:
//!   docker compose up mongo-shard-0 mongo-shard-1 -d
//!   cargo run --release --bin sharding-demo
//!   cargo run --release --bin sharding-demo -- --margin-arcsec 0  # without margins for comparison

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
#[command(name = "sharding-demo", about = "Two-machine HEALPix sharding demo with margin overlap")]
struct Args {
    #[arg(long, default_value = "mongodb://localhost:27081")]
    shard0_uri: String,
    #[arg(long, default_value = "mongodb://localhost:27082")]
    shard1_uri: String,
    /// Number of alerts to scatter across the sky
    #[arg(long, default_value_t = 100_000)]
    n_alerts: usize,
    /// Number of cone searches to run
    #[arg(long, default_value_t = 500)]
    n_queries: usize,
    /// Cone search radius in arcseconds
    #[arg(long, default_value_t = 10.0)]
    radius_arcsec: f64,
    /// Margin buffer size in arcseconds. Sources within this distance of
    /// a partition boundary are copied to the neighboring machine.
    /// Set to 0 to disable margins (shows cross-shard queries).
    /// Should be >= radius_arcsec for complete single-machine queries.
    #[arg(long, default_value_t = 30.0)]
    margin_arcsec: f64,
}

fn hpx_value(ra_deg: f64, dec_deg: f64, depth: u8) -> i64 {
    get(depth).hash(ra_deg.to_radians(), dec_deg.to_radians()) as i64
}

/// Primary machine: base cells 0–5 → machine 0, 6–11 → machine 1.
fn primary_machine(hpx29: i64) -> usize {
    let base_cell = hpx29 >> (2 * HPX_DEPTH as i64);
    if base_cell < N_BASE_CELLS / 2 { 0 } else { 1 }
}

/// Check if a point is within `margin_rad` of the boundary between the two
/// machine regions. We do this by checking if the cone around the point
/// overlaps base cells belonging to the OTHER machine.
fn near_boundary(ra_deg: f64, dec_deg: f64, margin_rad: f64, primary: usize) -> bool {
    if margin_rad <= 0.0 {
        return false;
    }
    let bmoc = nested::cone_coverage_approx(
        0, // depth 0 = 12 base cells
        ra_deg.to_radians(),
        dec_deg.to_radians(),
        margin_rad,
    );
    for r in bmoc.to_ranges().iter() {
        for cell in r.start..r.end {
            let cell_machine = if (cell as i64) < N_BASE_CELLS / 2 { 0 } else { 1 };
            if cell_machine != primary {
                return true; // cone touches the other machine's territory
            }
        }
    }
    false
}

/// Build HEALPix range filter on the `hpx` field.
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
    let margin_rad = (args.margin_arcsec / 3600.0_f64).to_radians();

    // --- Connect ---
    println!("Connecting...");
    let db0 = connect(&args.shard0_uri, "Machine 0").await?;
    let db1 = connect(&args.shard1_uri, "Machine 1").await?;
    let machines = [&db0, &db1];
    println!("  Machine 0: {} (base cells 0–5)", args.shard0_uri);
    println!("  Machine 1: {} (base cells 6–11)", args.shard1_uri);
    if args.margin_arcsec > 0.0 {
        println!("  Margin buffer: {}\" (sources near boundary duplicated to neighbor)", args.margin_arcsec);
    } else {
        println!("  Margin buffer: DISABLED (cross-shard queries will be needed)");
    }

    let coll_name = "alerts";
    for m in &machines {
        let _ = m.collection::<mongodb::bson::Document>(coll_name).drop().await;
    }
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    for m in &machines {
        m.collection::<mongodb::bson::Document>(coll_name)
            .create_index(mongodb::IndexModel::builder().keys(doc! { "hpx": 1 }).build())
            .await?;
    }

    // --- Insert alerts with margin overlap ---
    println!("\n=== Inserting {} alerts (margin={}\" ) ===\n", args.n_alerts, args.margin_arcsec);
    let mut rng = rand::rng();
    let mut primary_counts = [0usize; 2];
    let mut margin_counts = [0usize; 2]; // margin copies on each machine
    let batch_size = 5000;
    let mut batches: [Vec<mongodb::bson::Document>; 2] = [Vec::new(), Vec::new()];

    let t = Instant::now();
    for i in 0..args.n_alerts {
        let ra: f64 = rng.random::<f64>() * 360.0;
        let u: f64 = rng.random::<f64>();
        let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
        let h29 = hpx_value(ra, dec, HPX_DEPTH);
        let primary = primary_machine(h29);

        let alert_doc = doc! {
            "_id": format!("a{}", i),
            "ra": ra,
            "dec": dec,
            "hpx": h29,
            "is_margin": false,
        };

        batches[primary].push(alert_doc);
        primary_counts[primary] += 1;

        // If near boundary, also insert a margin copy on the OTHER machine
        if near_boundary(ra, dec, margin_rad, primary) {
            let other = 1 - primary;
            let margin_doc = doc! {
                "_id": format!("m{}", i),  // different _id to avoid conflict
                "ra": ra,
                "dec": dec,
                "hpx": h29,
                "is_margin": true,
            };
            batches[other].push(margin_doc);
            margin_counts[other] += 1;
        }

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

    let total_0 = primary_counts[0] + margin_counts[0];
    let total_1 = primary_counts[1] + margin_counts[1];
    let total_margin = margin_counts[0] + margin_counts[1];
    let margin_pct = total_margin as f64 / args.n_alerts as f64 * 100.0;

    println!("  Machine 0: {} primary + {} margin = {} total", primary_counts[0], margin_counts[0], total_0);
    println!("  Machine 1: {} primary + {} margin = {} total", primary_counts[1], margin_counts[1], total_1);
    println!("  Margin overhead: {} copies ({:.2}% of catalog)", total_margin, margin_pct);
    println!("  Insert time: {:.2}s", insert_time.as_secs_f64());

    // --- Storage ---
    println!("\nStorage per machine:");
    let mut size_tbl = Table::new();
    size_tbl.set_header(vec!["", "Machine 0", "Machine 1"]);
    let mut data_sizes = [0i64; 2];
    let mut idx_sz = [0i64; 2];
    for (i, m) in machines.iter().enumerate() {
        let coll = m.collection::<mongodb::bson::Document>(coll_name);
        let stats: Vec<_> = coll.aggregate(vec![doc! { "$collStats": { "storageStats": {} } }])
            .await?.collect::<Vec<_>>().await;
        if let Some(Ok(d)) = stats.first() {
            if let Ok(s) = d.get_document("storageStats") {
                let get = |k: &str| -> i64 {
                    match s.get(k) {
                        Some(mongodb::bson::Bson::Int64(v)) => *v,
                        Some(mongodb::bson::Bson::Int32(v)) => *v as i64,
                        Some(mongodb::bson::Bson::Double(v)) => *v as i64,
                        _ => 0,
                    }
                };
                data_sizes[i] = get("storageSize");
                idx_sz[i] = get("totalIndexSize");
            }
        }
    }
    size_tbl.add_row(vec![Cell::new("Data"), Cell::new(fmt_bytes(data_sizes[0])), Cell::new(fmt_bytes(data_sizes[1]))]);
    size_tbl.add_row(vec![Cell::new("Indexes"), Cell::new(fmt_bytes(idx_sz[0])), Cell::new(fmt_bytes(idx_sz[1]))]);
    size_tbl.add_row(vec![Cell::new("Total"), Cell::new(fmt_bytes(data_sizes[0]+idx_sz[0])), Cell::new(fmt_bytes(data_sizes[1]+idx_sz[1]))]);
    println!("{}", size_tbl);

    // --- Cone search ---
    println!("\n=== Cone search: {} queries at {}\" radius ===", args.n_queries, args.radius_arcsec);
    if args.margin_arcsec >= args.radius_arcsec {
        println!("  margin ({}\") >= radius ({}\") → all queries answered by single machine\n",
            args.margin_arcsec, args.radius_arcsec);
    } else if args.margin_arcsec > 0.0 {
        println!("  margin ({}\") < radius ({}\") → some boundary queries may miss results\n",
            args.margin_arcsec, args.radius_arcsec);
    } else {
        println!("  NO margin → boundary queries must hit both machines\n");
    }

    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(args.n_queries);
    for _ in 0..args.n_queries {
        let ra: f64 = rng.random::<f64>() * 360.0;
        let u: f64 = rng.random::<f64>();
        let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
        positions.push((ra, dec));
    }

    let mut queries_single = 0usize;
    let mut queries_both = 0usize;
    let mut total_exact_hits = 0usize;
    let mut total_machines_queried = 0usize;

    let t = Instant::now();
    for &(ra, dec) in &positions {
        let h29 = hpx_value(ra, dec, HPX_DEPTH);
        let primary = primary_machine(h29);

        if args.margin_arcsec >= args.radius_arcsec {
            // Margin is large enough — always query only the primary machine.
            // The margin buffer ensures all neighbors are present.
            total_machines_queried += 1;
            queries_single += 1;

            let (filter, _) = hpx_range_filter(ra, dec, radius_rad);
            let coll = machines[primary].collection::<mongodb::bson::Document>(coll_name);
            let mut cursor = coll.find(filter).await?;
            while let Some(r) = cursor.next().await {
                let doc = r?;
                let src_ra = doc.get_f64("ra").unwrap_or(0.0);
                let src_dec = doc.get_f64("dec").unwrap_or(0.0);
                if angular_distance_arcsec(ra, dec, src_ra, src_dec) <= args.radius_arcsec {
                    total_exact_hits += 1;
                }
            }
        } else {
            // No margin or insufficient margin — check if we need both machines
            let shard_bmoc = nested::cone_coverage_approx(
                0, ra.to_radians(), dec.to_radians(), radius_rad,
            );
            let target_cells: Vec<i64> = shard_bmoc.to_ranges().iter()
                .flat_map(|r| (r.start..r.end).map(|x| x as i64))
                .collect();

            let mut need = [false; 2];
            for cell in &target_cells {
                if *cell < N_BASE_CELLS / 2 { need[0] = true; } else { need[1] = true; }
            }
            if need[0] && need[1] { queries_both += 1; } else { queries_single += 1; }

            let (filter, _) = hpx_range_filter(ra, dec, radius_rad);
            for (idx, needed) in need.iter().enumerate() {
                if !needed { continue; }
                total_machines_queried += 1;
                let coll = machines[idx].collection::<mongodb::bson::Document>(coll_name);
                let mut cursor = coll.find(filter.clone()).await?;
                while let Some(r) = cursor.next().await {
                    let doc = r?;
                    let src_ra = doc.get_f64("ra").unwrap_or(0.0);
                    let src_dec = doc.get_f64("dec").unwrap_or(0.0);
                    if angular_distance_arcsec(ra, dec, src_ra, src_dec) <= args.radius_arcsec {
                        total_exact_hits += 1;
                    }
                }
            }
        }
    }
    let query_time = t.elapsed();

    let avg_machines = total_machines_queried as f64 / args.n_queries as f64;

    println!("Query routing:");
    println!("  Single-machine queries: {} ({:.0}%)", queries_single, queries_single as f64 / args.n_queries as f64 * 100.0);
    println!("  Cross-shard queries:    {} ({:.0}%)", queries_both, queries_both as f64 / args.n_queries as f64 * 100.0);
    println!("  Avg machines per query: {:.2}", avg_machines);
    println!();
    println!("Results:");
    println!("  Exact hits: {}", total_exact_hits);
    println!("  Query time: {:.2}s ({:.0} queries/sec)",
        query_time.as_secs_f64(),
        args.n_queries as f64 / query_time.as_secs_f64());

    // Key result
    println!("\n=== Key result ===");
    if args.margin_arcsec >= args.radius_arcsec {
        println!("  With {}\" margin, 100% of queries answered by a SINGLE machine.", args.margin_arcsec);
        println!("  No cross-shard communication needed. Each machine is self-sufficient.");
        println!("  Margin storage overhead: {:.2}% ({} duplicated sources).", margin_pct, total_margin);
    } else if queries_both > 0 {
        println!("  Without sufficient margin, {} queries ({:.1}%) needed both machines.",
            queries_both, queries_both as f64 / args.n_queries as f64 * 100.0);
    } else {
        println!("  All queries fit within a single base cell at this radius.");
    }

    // ===================================================================
    // Object ID lookup demo
    // ===================================================================
    // When data is sharded by sky position, how do you find an alert by
    // its objectId? You don't know which machine it's on.
    //
    // Solution: a lightweight lookup index that maps objectId → machine.
    // This lives on ONE machine (or is replicated to all) and is tiny —
    // just the objectId and shard assignment.

    println!("\n=== Object ID lookup demo ===\n");

    // Clear alert collections from the cone search demo
    for m in &machines {
        let _ = m.collection::<mongodb::bson::Document>(coll_name).drop().await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    for m in &machines {
        m.collection::<mongodb::bson::Document>(coll_name)
            .create_index(mongodb::IndexModel::builder().keys(doc! { "objectId": 1 }).build())
            .await?;
    }

    let lookup_coll_name = "object_lookup";

    // Build the lookup index on machine 0 (could be any machine or a
    // dedicated lightweight instance). For each alert, store just the
    // objectId and which machine owns it.
    println!("Building objectId → machine lookup index...");
    let lookup_coll = machines[0].collection::<mongodb::bson::Document>(lookup_coll_name);
    let _ = lookup_coll.drop().await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    lookup_coll.create_index(
        mongodb::IndexModel::builder().keys(doc! { "oid": 1 }).options(
            mongodb::options::IndexOptions::builder().unique(true).build()
        ).build()
    ).await?;

    // Re-insert alerts and populate the lookup index at the same time.
    // In production this happens during ingest — one extra write per alert.
    let mut lookup_batch: Vec<mongodb::bson::Document> = Vec::new();
    let mut alert_batches: [Vec<mongodb::bson::Document>; 2] = [Vec::new(), Vec::new()];
    let n_lookup = 50_000usize;

    let t = Instant::now();
    for i in 0..n_lookup {
        let ra: f64 = rng.random::<f64>() * 360.0;
        let u: f64 = rng.random::<f64>();
        let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
        let h29 = hpx_value(ra, dec, HPX_DEPTH);
        let primary = primary_machine(h29);
        let object_id = format!("ZTF24{:07x}", i);

        alert_batches[primary].push(doc! {
            "_id": format!("a{}", i),
            "objectId": &object_id,
            "ra": ra,
            "dec": dec,
            "hpx": h29,
        });

        lookup_batch.push(doc! {
            "oid": &object_id,
            "machine": primary as i32,
        });

        // Flush
        for m_idx in 0..2 {
            if alert_batches[m_idx].len() >= 5000 {
                let batch = std::mem::take(&mut alert_batches[m_idx]);
                machines[m_idx].collection::<mongodb::bson::Document>(coll_name)
                    .insert_many(batch).await?;
            }
        }
        if lookup_batch.len() >= 5000 {
            let batch = std::mem::take(&mut lookup_batch);
            lookup_coll.insert_many(batch).await?;
        }
    }
    // Flush remaining
    for m_idx in 0..2 {
        if !alert_batches[m_idx].is_empty() {
            let batch = std::mem::take(&mut alert_batches[m_idx]);
            machines[m_idx].collection::<mongodb::bson::Document>(coll_name)
                .insert_many(batch).await?;
        }
    }
    if !lookup_batch.is_empty() {
        lookup_coll.insert_many(lookup_batch).await?;
    }
    let lookup_build_time = t.elapsed();

    // Measure lookup index size
    let lookup_stats: Vec<_> = lookup_coll.aggregate(vec![
        doc! { "$collStats": { "storageStats": {} } }
    ]).await?.collect::<Vec<_>>().await;
    let lookup_size = if let Some(Ok(doc)) = lookup_stats.first() {
        if let Ok(s) = doc.get_document("storageStats") {
            let data = match s.get("storageSize") {
                Some(mongodb::bson::Bson::Int64(v)) => *v,
                Some(mongodb::bson::Bson::Int32(v)) => *v as i64,
                Some(mongodb::bson::Bson::Double(v)) => *v as i64,
                _ => 0,
            };
            let idx = match s.get("totalIndexSize") {
                Some(mongodb::bson::Bson::Int64(v)) => *v,
                Some(mongodb::bson::Bson::Int32(v)) => *v as i64,
                Some(mongodb::bson::Bson::Double(v)) => *v as i64,
                _ => 0,
            };
            data + idx
        } else { 0 }
    } else { 0 };

    println!("  {} objects indexed in {:.2}s", n_lookup, lookup_build_time.as_secs_f64());
    println!("  Lookup index size: {} ({:.0} bytes/object)",
        fmt_bytes(lookup_size), lookup_size as f64 / n_lookup as f64);
    println!("  At 100M objects: ~{:.0} MB",
        lookup_size as f64 / n_lookup as f64 * 100_000_000.0 / 1024.0 / 1024.0);

    // Now demonstrate: query by objectId using the lookup
    println!("\nQuerying {} random objects by ID...", args.n_queries);
    let n_id_queries = args.n_queries.min(500);
    let mut id_found = 0usize;

    let t = Instant::now();
    for i in 0..n_id_queries {
        // Pick a random objectId
        let target_idx = rng.random_range(0..n_lookup);
        let target_oid = format!("ZTF24{:07x}", target_idx);

        // Step 1: lookup which machine has this object
        let lookup_result = lookup_coll
            .find_one(doc! { "oid": &target_oid })
            .projection(doc! { "machine": 1, "_id": 0 })
            .await?;

        if let Some(lookup_doc) = lookup_result {
            let machine_idx = lookup_doc.get_i32("machine").unwrap_or(0) as usize;

            // Step 2: query ONLY that machine
            let alert = machines[machine_idx]
                .collection::<mongodb::bson::Document>(coll_name)
                .find_one(doc! { "objectId": &target_oid })
                .await?;

            if alert.is_some() {
                id_found += 1;
            }
        }
    }
    let id_query_time = t.elapsed();

    println!("  Found: {}/{} ({:.0}%)", id_found, n_id_queries,
        id_found as f64 / n_id_queries as f64 * 100.0);
    println!("  Time: {:.2}s ({:.0} lookups/sec)",
        id_query_time.as_secs_f64(),
        n_id_queries as f64 / id_query_time.as_secs_f64());
    println!("  Avg: {:.2}ms per lookup (including routing)",
        id_query_time.as_secs_f64() / n_id_queries as f64 * 1000.0);

    // Compare: scatter-gather (query both machines)
    println!("\nCompare: scatter-gather (query ALL machines, no lookup)...");
    let t = Instant::now();
    let mut scatter_found = 0usize;
    for _ in 0..n_id_queries {
        let target_idx = rng.random_range(0..n_lookup);
        let target_oid = format!("ZTF24{:07x}", target_idx);

        for m in &machines {
            let result = m.collection::<mongodb::bson::Document>(coll_name)
                .find_one(doc! { "objectId": &target_oid })
                .await?;
            if result.is_some() {
                scatter_found += 1;
            }
        }
    }
    let scatter_time = t.elapsed();

    println!("  Found: {}/{}", scatter_found, n_id_queries);
    println!("  Time: {:.2}s ({:.0} lookups/sec)",
        scatter_time.as_secs_f64(),
        n_id_queries as f64 / scatter_time.as_secs_f64());
    println!("  Avg: {:.2}ms per lookup",
        scatter_time.as_secs_f64() / n_id_queries as f64 * 1000.0);

    let speedup = scatter_time.as_secs_f64() / id_query_time.as_secs_f64();
    println!("\n  Lookup-routed is {:.1}x faster than scatter-gather.", speedup);

    // Cleanup
    let _ = lookup_coll.drop().await;
    for m in &machines {
        let _ = m.collection::<mongodb::bson::Document>(coll_name).drop().await;
    }
    println!("\nCollections dropped.");

    Ok(())
}
