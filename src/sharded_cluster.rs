//! MongoDB native sharding demo with HEALPix shard key.
//!
//! Connects through mongos (the MongoDB router) and lets MongoDB handle
//! shard routing automatically. Compares:
//!   - Unsharded collection (single mongos, all data on one shard)
//!   - Sharded collection with hpx_16 as range shard key
//!
//! This tests the question: does MongoDB's native sharding with a HEALPix
//! shard key give you automatic spatial locality and parallel query execution?
//!
//! Usage:
//!   docker compose -f docker-compose.sharded.yaml up -d
//!   # Wait for init container to finish (~20s)
//!   cargo run --release --bin sharded-cluster

use cdshealpix::nested::{self, get};
use clap::Parser;
use comfy_table::{Cell, Table};
use futures::stream::StreamExt;
use mongodb::bson::doc;
use mongodb::options::ClientOptions;
use rand::Rng;
use std::time::Instant;

const HPX_DEPTH: u8 = 29;
const SHARD_DEPTH: u8 = 16;
const QUERY_DEPTH: u8 = 16;

#[derive(Parser)]
#[command(name = "sharded-cluster", about = "MongoDB native sharding with HEALPix shard key")]
struct Args {
    /// Mongos URI (the router, not individual shards)
    #[arg(long, default_value = "mongodb://localhost:27100")]
    mongos_uri: String,
    /// Number of documents to insert
    #[arg(long, default_value_t = 200_000)]
    catalog_size: usize,
    /// Number of queries per radius
    #[arg(long, default_value_t = 500)]
    n_queries: usize,
}

fn hpx_value(ra_deg: f64, dec_deg: f64, depth: u8) -> i64 {
    get(depth).hash(ra_deg.to_radians(), dec_deg.to_radians()) as i64
}

fn hpx_range_filter(ra: f64, dec: f64, radius_rad: f64) -> mongodb::bson::Document {
    let bmoc = nested::cone_coverage_approx(
        QUERY_DEPTH, ra.to_radians(), dec.to_radians(), radius_rad,
    );
    let ranges = bmoc.to_ranges();
    let shift = 2 * (HPX_DEPTH - QUERY_DEPTH);
    if ranges.len() == 1 {
        doc! { "hpx": { "$gte": (ranges[0].start << shift) as i64, "$lt": (ranges[0].end << shift) as i64 } }
    } else {
        let clauses: Vec<mongodb::bson::Document> = ranges.iter().map(|r| {
            doc! { "hpx": { "$gte": (r.start << shift) as i64, "$lt": (r.end << shift) as i64 } }
        }).collect();
        doc! { "$or": clauses }
    }
}

fn geojson_filter(ra: f64, dec: f64, radius_rad: f64) -> mongodb::bson::Document {
    doc! { "loc": { "$geoWithin": { "$centerSphere": [[ra - 180.0, dec], radius_rad] } } }
}

fn angular_distance_arcsec(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (ra1, dec1) = (ra1.to_radians(), dec1.to_radians());
    let (ra2, dec2) = (ra2.to_radians(), dec2.to_radians());
    let dlat = dec2 - dec1;
    let dlon = ra2 - ra1;
    let a = (dlat / 2.0).sin().powi(2) + dec1.cos() * dec2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().asin().to_degrees() * 3600.0
}

fn bson_to_i64(val: Option<&mongodb::bson::Bson>) -> i64 {
    match val {
        Some(mongodb::bson::Bson::Int64(v)) => *v,
        Some(mongodb::bson::Bson::Int32(v)) => *v as i64,
        Some(mongodb::bson::Bson::Double(v)) => *v as i64,
        _ => 0,
    }
}

fn fmt_bytes(bytes: i64) -> String {
    if bytes >= 1024 * 1024 { format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0) }
    else if bytes >= 1024 { format!("{:.1} KB", bytes as f64 / 1024.0) }
    else { format!("{} B", bytes) }
}

fn random_positions(rng: &mut impl Rng, n: usize) -> Vec<(f64, f64)> {
    (0..n).map(|_| {
        let ra: f64 = rng.random::<f64>() * 360.0;
        let u: f64 = rng.random::<f64>();
        let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
        (ra, dec)
    }).collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut opts = ClientOptions::parse(&args.mongos_uri).await?;
    opts.server_selection_timeout = Some(std::time::Duration::from_secs(30));
    let client = mongodb::Client::with_options(opts)?;
    let db = client.database("boom_sharding_test");
    db.run_command(doc! { "ping": 1 }).await?;
    println!("Connected to mongos at {}", args.mongos_uri);

    // Check shard status
    let admin = client.database("admin");
    let status = admin.run_command(doc! { "listShards": 1 }).await?;
    if let Ok(shards) = status.get_array("shards") {
        println!("Cluster has {} shards:", shards.len());
        for s in shards {
            if let Some(doc) = s.as_document() {
                println!("  {} → {}", doc.get_str("_id").unwrap_or("?"), doc.get_str("host").unwrap_or("?"));
            }
        }
    }

    let coll_name = "catalog";
    let _ = db.collection::<mongodb::bson::Document>(coll_name).drop().await;
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // --- Enable sharding on the database and collection ---
    println!("\nSetting up sharded collection...");

    // Enable sharding on the database
    let _ = admin.run_command(doc! { "enableSharding": "boom_sharding_test" }).await;

    // Create the collection and hpx index (required before shardCollection)
    let coll = db.collection::<mongodb::bson::Document>(coll_name);
    coll.create_index(
        mongodb::IndexModel::builder().keys(doc! { "hpx": 1 }).build()
    ).await?;

    // Shard the collection using hpx as range shard key
    match admin.run_command(doc! {
        "shardCollection": "boom_sharding_test.catalog",
        "key": { "hpx": 1 },
    }).await {
        Ok(_) => println!("  Collection sharded on hpx (range-based)"),
        Err(e) => println!("  Shard command: {} (may already be sharded)", e),
    }

    // Also create 2dsphere index for comparison queries
    coll.create_index(
        mongodb::IndexModel::builder().keys(doc! { "loc": "2dsphere" }).build()
    ).await?;

    // Pre-split chunks at HEALPix depth-4 boundaries (192 chunks)
    // so data distributes across shards from the start
    println!("  Pre-splitting chunks at depth-4 boundaries...");
    let shift = 2 * (HPX_DEPTH - 4);
    let n_pixels_d4 = 12 * 4u32.pow(4); // 3072
    let mut splits_ok = 0;
    for p in 1..n_pixels_d4 {
        let split_point = (p as i64) << shift;
        match admin.run_command(doc! {
            "split": "boom_sharding_test.catalog",
            "middle": { "hpx": split_point },
        }).await {
            Ok(_) => splits_ok += 1,
            Err(_) => {},
        }
    }
    println!("  Created {} chunk split points", splits_ok);

    // --- Insert data ---
    println!("\n=== Inserting {} documents ===\n", args.catalog_size);
    let mut rng = rand::rng();
    let batch_size = 5000;

    let t = Instant::now();
    let mut i = 0usize;
    while i < args.catalog_size {
        let end = (i + batch_size).min(args.catalog_size);
        let mut batch = Vec::with_capacity(end - i);
        for j in i..end {
            let ra: f64 = rng.random::<f64>() * 360.0;
            let u: f64 = rng.random::<f64>();
            let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
            batch.push(doc! {
                "_id": j as i64,
                "ra": ra,
                "dec": dec,
                "hpx": hpx_value(ra, dec, HPX_DEPTH),
                "loc": { "type": "Point", "coordinates": [ra - 180.0, dec] },
            });
        }
        coll.insert_many(batch).await?;
        i = end;
    }
    let insert_time = t.elapsed();
    println!("  Insert: {:.2}s ({:.0} docs/sec)", insert_time.as_secs_f64(),
        args.catalog_size as f64 / insert_time.as_secs_f64());

    // Show chunk distribution
    let config_db = client.database("config");
    let chunks_coll = config_db.collection::<mongodb::bson::Document>("chunks");
    let mut shard_chunks: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut cursor = chunks_coll.find(doc! { "ns": "boom_sharding_test.catalog" }).await?;
    while let Some(Ok(chunk)) = cursor.next().await {
        let shard = chunk.get_str("shard").unwrap_or("?").to_string();
        *shard_chunks.entry(shard).or_insert(0) += 1;
    }
    println!("\nChunk distribution:");
    for (shard, count) in &shard_chunks {
        println!("  {}: {} chunks", shard, count);
    }

    // --- Benchmark queries ---
    let radii_arcsec = vec![2.0, 10.0, 30.0, 60.0, 300.0];

    let mut tbl = Table::new();
    tbl.set_header(vec!["Radius", "Method", "Time (s)", "Hits"]);

    println!("\n=== Query benchmark: {} queries/radius ===\n", args.n_queries);

    for radius_arcsec in &radii_arcsec {
        let radius_rad = (*radius_arcsec / 3600.0_f64).to_radians();

        // 2dsphere through mongos
        let positions = random_positions(&mut rng, args.n_queries);
        let mut geo_hits = 0usize;
        let t = Instant::now();
        for &(ra, dec) in &positions {
            let mut cursor = coll.find(geojson_filter(ra, dec, radius_rad)).await?;
            while let Some(r) = cursor.next().await { let _ = r?; geo_hits += 1; }
        }
        let geo_time = t.elapsed();

        // HEALPix through mongos (mongos routes based on shard key in filter)
        let positions = random_positions(&mut rng, args.n_queries);
        let mut hpx_hits = 0usize;
        let mut hpx_exact = 0usize;
        let t = Instant::now();
        for &(ra, dec) in &positions {
            let filter = hpx_range_filter(ra, dec, radius_rad);
            let mut cursor = coll.find(filter).await?;
            while let Some(r) = cursor.next().await {
                let doc = r?;
                hpx_hits += 1;
                let src_ra = doc.get_f64("ra").unwrap_or(0.0);
                let src_dec = doc.get_f64("dec").unwrap_or(0.0);
                if angular_distance_arcsec(ra, dec, src_ra, src_dec) <= *radius_arcsec {
                    hpx_exact += 1;
                }
            }
        }
        let hpx_time = t.elapsed();

        let speedup = geo_time.as_secs_f64() / hpx_time.as_secs_f64();

        tbl.add_row(vec![
            Cell::new(format!("{}\"", radius_arcsec)),
            Cell::new("2dsphere (via mongos)"),
            Cell::new(format!("{:.3}", geo_time.as_secs_f64())),
            Cell::new(geo_hits),
        ]);
        tbl.add_row(vec![
            Cell::new(""),
            Cell::new("HEALPix (via mongos)"),
            Cell::new(format!("{:.3}", hpx_time.as_secs_f64())),
            Cell::new(format!("{} raw / {} exact", hpx_hits, hpx_exact)),
        ]);
        tbl.add_row(vec![
            Cell::new(""),
            Cell::new("Speedup"),
            Cell::new(format!("{:.1}x", speedup)),
            Cell::new(""),
        ]);
    }

    println!("{}", tbl);

    println!("\nKey question: does the HEALPix shard key let mongos route queries");
    println!("to only the relevant shard(s)? If so, HEALPix queries should be faster");
    println!("than 2dsphere queries (which mongos must scatter to all shards).");

    // Cleanup
    let _ = db.collection::<mongodb::bson::Document>(coll_name).drop().await;
    println!("\nCollection dropped.");

    Ok(())
}
