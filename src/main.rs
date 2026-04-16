//! Benchmark: HEALPix integer range queries vs MongoDB 2dsphere for cone searches.
//!
//! Standalone benchmarking tool — no boom dependency.
//! Tests three query strategies:
//!   1. **2dsphere**: MongoDB's native $geoWithin/$centerSphere
//!   2. **HEALPix ranges**: $or with $gte/$lt ranges on depth-29 hpx (boom approach)
//!   3. **HEALPix $in**: $in with pixel list at query depth (extcats approach)
//!
//! All HEALPix results are post-filtered for exact distance (the right approach
//! per extcats — HEALPix is the coarse filter, distance check is the fine filter).
//!
//! Usage:
//!   docker compose up -d
//!   cargo run --release -- --catalog-size 1000000 --n-queries 1000
//!   cargo run --release -- --explain   # show MongoDB query plans

use cdshealpix::nested::{self, get};
use clap::Parser;
use comfy_table::{Cell, Table};
use futures::stream::StreamExt;
use mongodb::bson::doc;
use mongodb::options::ClientOptions;
use rand::Rng;
use std::time::Instant;

/// HEALPix depth 29 — matches healpix-alchemy's HPX_MAX_ORDER.
const HPX_DEPTH: u8 = 29;

/// Default query depth for the $in approach (order 16 = extcats standard, ~3.2" pixels).
const DEFAULT_IN_DEPTH: u8 = 16;

#[derive(Parser)]
#[command(name = "boom-healpix", about = "Benchmark HEALPix vs 2dsphere cone searches")]
struct Args {
    /// MongoDB connection URI
    #[arg(long, default_value = "mongodb://localhost:27099")]
    mongo_uri: String,

    /// Number of random sky points to insert
    #[arg(long, default_value_t = 1_000_000)]
    catalog_size: usize,

    /// Number of cone search queries per radius
    #[arg(long, default_value_t = 1000)]
    n_queries: usize,

    /// Keep the collection after the benchmark (don't drop it)
    #[arg(long, default_value_t = false)]
    keep: bool,

    /// Show MongoDB explain output for the first query at each radius
    #[arg(long, default_value_t = false)]
    explain: bool,

    /// HEALPix depth for $in queries (default: 16, like extcats)
    #[arg(long, default_value_t = DEFAULT_IN_DEPTH)]
    in_depth: u8,

    /// Write results to JSON file (for plotting)
    #[arg(long)]
    output_json: Option<String>,
}

// ---------------------------------------------------------------------------
// HEALPix helpers
// ---------------------------------------------------------------------------

/// Compute HEALPix NESTED index at the given depth.
fn hpx(ra_deg: f64, dec_deg: f64, depth: u8) -> u64 {
    get(depth).hash(ra_deg.to_radians(), dec_deg.to_radians())
}

/// Choose query depth for the $or-ranges approach.
/// Targets pixel_side ≈ radius so the cone spans ~4 pixels across.
fn auto_query_depth(radius_rad: f64) -> u8 {
    let target = 2.0 * std::f64::consts::PI / (3.0_f64.sqrt() * radius_rad);
    let d = target.log2().floor() as i32;
    d.clamp(0, HPX_DEPTH as i32) as u8
}

/// Build MongoDB filter using $or with HEALPix ranges (boom approach).
/// Returns (filter_doc, number_of_ranges).
fn hpx_range_filter(
    ra_deg: f64,
    dec_deg: f64,
    radius_rad: f64,
    query_depth: u8,
) -> (mongodb::bson::Document, usize) {
    let bmoc = nested::cone_coverage_approx(
        query_depth,
        ra_deg.to_radians(),
        dec_deg.to_radians(),
        radius_rad,
    );
    let ranges = bmoc.to_ranges();
    let n = ranges.len();
    let shift = 2 * (HPX_DEPTH - query_depth);

    let filter = if n == 1 {
        doc! {
            "hpx": {
                "$gte": (ranges[0].start << shift) as i64,
                "$lt": (ranges[0].end << shift) as i64,
            }
        }
    } else {
        let or_clauses: Vec<mongodb::bson::Document> = ranges
            .iter()
            .map(|r| {
                doc! {
                    "hpx": {
                        "$gte": (r.start << shift) as i64,
                        "$lt": (r.end << shift) as i64,
                    }
                }
            })
            .collect();
        doc! { "$or": or_clauses }
    };

    (filter, n)
}

/// Build MongoDB filter using $in with flat pixel list (extcats approach).
/// Queries the `hpx_{depth}` field with the covering pixel set.
/// Returns (filter_doc, number_of_pixels, field_name).
fn hpx_in_filter(
    ra_deg: f64,
    dec_deg: f64,
    radius_rad: f64,
    query_depth: u8,
) -> (mongodb::bson::Document, usize) {
    let bmoc = nested::cone_coverage_approx(
        query_depth,
        ra_deg.to_radians(),
        dec_deg.to_radians(),
        radius_rad,
    );

    // Flatten ranges to individual pixel IDs at query_depth
    let ranges = bmoc.to_ranges();
    let mut pixels: Vec<i64> = Vec::new();
    for r in ranges.iter() {
        for p in r.start..r.end {
            pixels.push(p as i64);
        }
    }

    let field = format!("hpx_{}", query_depth);
    let n = pixels.len();
    let filter = doc! { &field: { "$in": pixels } };
    (filter, n)
}

/// Build 2dsphere filter.
fn geojson_filter(ra_deg: f64, dec_deg: f64, radius_rad: f64) -> mongodb::bson::Document {
    doc! {
        "loc": {
            "$geoWithin": {
                "$centerSphere": [[ra_deg - 180.0, dec_deg], radius_rad]
            }
        }
    }
}

/// Haversine distance in arcseconds.
fn angular_distance_arcsec(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let ra1 = ra1.to_radians();
    let dec1 = dec1.to_radians();
    let ra2 = ra2.to_radians();
    let dec2 = dec2.to_radians();
    let dlat = dec2 - dec1;
    let dlon = ra2 - ra1;
    let a = (dlat / 2.0).sin().powi(2) + dec1.cos() * dec2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().asin().to_degrees() * 3600.0
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Extract an integer from a BSON value, regardless of whether MongoDB
/// sent it as Int32, Int64, or Double.
fn bson_to_i64(val: Option<&mongodb::bson::Bson>) -> i64 {
    match val {
        Some(mongodb::bson::Bson::Int64(v)) => *v,
        Some(mongodb::bson::Bson::Int32(v)) => *v as i64,
        Some(mongodb::bson::Bson::Double(v)) => *v as i64,
        _ => 0,
    }
}

fn fmt_bytes(bytes: i64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn fmt_rate(count: usize, secs: f64) -> String {
    let rate = count as f64 / secs;
    if rate > 1000.0 { format!("{:.0}K/s", rate / 1000.0) }
    else { format!("{:.0}/s", rate) }
}

// ---------------------------------------------------------------------------
// Explain helper
// ---------------------------------------------------------------------------

async fn explain_query(
    db: &mongodb::Database,
    coll_name: &str,
    filter: &mongodb::bson::Document,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = db
        .run_command(doc! {
            "explain": { "find": coll_name, "filter": filter },
            "verbosity": "executionStats",
        })
        .await?;

    if let Ok(exec_stats) = result.get_document("executionStats") {
        let examined = exec_stats.get_i64("totalKeysExamined").unwrap_or(0);
        let docs_examined = exec_stats.get_i64("totalDocsExamined").unwrap_or(0);
        let returned = exec_stats.get_i64("nReturned").unwrap_or(0);
        let millis = exec_stats.get_i64("executionTimeMillis").unwrap_or(0);
        println!(
            "  {:<14} keys={:<8} docs={:<8} returned={:<6} {}ms",
            format!("{}:", label), examined, docs_examined, returned, millis
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Benchmark a single method
// ---------------------------------------------------------------------------

struct BenchResult {
    total_time: std::time::Duration,
    raw_hits: usize,     // before post-filter
    exact_hits: usize,   // after post-filter
}

async fn bench_geojson(
    coll: &mongodb::Collection<mongodb::bson::Document>,
    positions: &[(f64, f64)],
    radius_arcsec: f64,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let radius_rad = (radius_arcsec / 3600.0).to_radians();
    let mut raw_hits = 0;
    let t = Instant::now();
    for &(ra, dec) in positions {
        let filter = geojson_filter(ra, dec, radius_rad);
        let mut cursor = coll.find(filter).await?;
        while let Some(r) = cursor.next().await {
            let _ = r?;
            raw_hits += 1;
        }
    }
    Ok(BenchResult { total_time: t.elapsed(), raw_hits, exact_hits: raw_hits })
}

async fn bench_hpx_ranges(
    coll: &mongodb::Collection<mongodb::bson::Document>,
    positions: &[(f64, f64)],
    radius_arcsec: f64,
    query_depth: u8,
) -> Result<(BenchResult, usize), Box<dyn std::error::Error>> {
    let radius_rad = (radius_arcsec / 3600.0).to_radians();
    let (_, sample_ranges) = hpx_range_filter(positions[0].0, positions[0].1, radius_rad, query_depth);

    let mut raw_hits = 0;
    let mut exact_hits = 0;
    let t = Instant::now();
    for &(ra, dec) in positions {
        let (filter, _) = hpx_range_filter(ra, dec, radius_rad, query_depth);
        let mut cursor = coll.find(filter).await?;
        while let Some(r) = cursor.next().await {
            let doc = r?;
            raw_hits += 1;
            let src_ra = doc.get_f64("ra").unwrap_or(0.0);
            let src_dec = doc.get_f64("dec").unwrap_or(0.0);
            if angular_distance_arcsec(ra, dec, src_ra, src_dec) <= radius_arcsec {
                exact_hits += 1;
            }
        }
    }
    Ok((BenchResult { total_time: t.elapsed(), raw_hits, exact_hits }, sample_ranges))
}

async fn bench_hpx_in(
    coll: &mongodb::Collection<mongodb::bson::Document>,
    positions: &[(f64, f64)],
    radius_arcsec: f64,
    query_depth: u8,
) -> Result<(BenchResult, usize), Box<dyn std::error::Error>> {
    let radius_rad = (radius_arcsec / 3600.0).to_radians();
    let (_, sample_npix) = hpx_in_filter(positions[0].0, positions[0].1, radius_rad, query_depth);

    let mut raw_hits = 0;
    let mut exact_hits = 0;
    let t = Instant::now();
    for &(ra, dec) in positions {
        let (filter, _) = hpx_in_filter(ra, dec, radius_rad, query_depth);
        let mut cursor = coll.find(filter).await?;
        while let Some(r) = cursor.next().await {
            let doc = r?;
            raw_hits += 1;
            let src_ra = doc.get_f64("ra").unwrap_or(0.0);
            let src_dec = doc.get_f64("dec").unwrap_or(0.0);
            if angular_distance_arcsec(ra, dec, src_ra, src_dec) <= radius_arcsec {
                exact_hits += 1;
            }
        }
    }
    Ok((BenchResult { total_time: t.elapsed(), raw_hits, exact_hits }, sample_npix))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // --- Connect ---
    let mut opts = ClientOptions::parse(&args.mongo_uri).await?;
    opts.server_selection_timeout = Some(std::time::Duration::from_secs(30));
    opts.connect_timeout = Some(std::time::Duration::from_secs(30));
    let client = mongodb::Client::with_options(opts)?;
    let db = client.database("boom_healpix_bench");
    db.run_command(doc! { "ping": 1 }).await?;
    println!("Connected to {}", args.mongo_uri);

    let coll_name = "catalog";
    let coll = db.collection::<mongodb::bson::Document>(coll_name);
    let _ = coll.drop().await;
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // --- Generate and insert catalog ---
    println!("\n=== Catalog: {} points ===\n", args.catalog_size);
    let mut rng = rand::rng();
    let batch_size = 5_000;

    let in_field = format!("hpx_{}", args.in_depth);

    print!("Inserting... ");
    let t = Instant::now();
    let mut i = 0usize;
    while i < args.catalog_size {
        let end = (i + batch_size).min(args.catalog_size);
        let mut batch = Vec::with_capacity(end - i);
        for j in i..end {
            let ra: f64 = rng.random::<f64>() * 360.0;
            let u: f64 = rng.random::<f64>();
            let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
            let h29 = hpx(ra, dec, HPX_DEPTH) as i64;
            let h_in = hpx(ra, dec, args.in_depth) as i64;
            batch.push(doc! {
                "_id": j as i64,
                "ra": ra,
                "dec": dec,
                "hpx": h29,
                &in_field: h_in,
                "loc": { "type": "Point", "coordinates": [ra - 180.0, dec] },
            });
        }
        coll.insert_many(batch).await?;
        i = end;
    }
    let insert_time = t.elapsed();
    println!("{:.2}s ({})", insert_time.as_secs_f64(), fmt_rate(args.catalog_size, insert_time.as_secs_f64()));

    // --- Create indexes ---
    println!("\nIndex build time:");
    let t = Instant::now();
    coll.create_index(mongodb::IndexModel::builder().keys(doc! { "loc": "2dsphere" }).build()).await?;
    let geo_t = t.elapsed();
    println!("  2dsphere:       {:.2}s", geo_t.as_secs_f64());

    let t = Instant::now();
    coll.create_index(mongodb::IndexModel::builder().keys(doc! { "hpx": 1 }).build()).await?;
    let hpx_t = t.elapsed();
    println!("  hpx (depth 29): {:.2}s", hpx_t.as_secs_f64());

    let t = Instant::now();
    coll.create_index(mongodb::IndexModel::builder().keys(doc! { &in_field: 1 }).build()).await?;
    let in_t = t.elapsed();
    println!("  {} (depth {}):  {:.2}s", in_field, args.in_depth, in_t.as_secs_f64());

    // --- Measure sizes ---
    println!("\nDisk usage:");
    let stats: Vec<_> = coll.aggregate(vec![doc! { "$collStats": { "storageStats": {} } }]).await?.collect::<Vec<_>>().await;
    if let Some(Ok(stats_doc)) = stats.first() {
        if let Ok(storage) = stats_doc.get_document("storageStats") {
            let data_size = bson_to_i64(storage.get("size"));
            let storage_size = bson_to_i64(storage.get("storageSize"));
            let total_idx = bson_to_i64(storage.get("totalIndexSize"));
            println!("  Data (uncompressed): {}", fmt_bytes(data_size));
            println!("  Data (on disk):      {}", fmt_bytes(storage_size));
            println!("  Total indexes:       {}", fmt_bytes(total_idx));
            println!("  Total footprint:     {}", fmt_bytes(storage_size + total_idx));
            if let Ok(idx_sizes) = storage.get_document("indexSizes") {
                let mut tbl = Table::new();
                tbl.set_header(vec!["Index", "Size", "Per doc"]);
                for (key, val) in idx_sizes {
                    let bytes = bson_to_i64(Some(val));
                    tbl.add_row(vec![
                        Cell::new(key),
                        Cell::new(fmt_bytes(bytes)),
                        Cell::new(format!("{:.0} B", bytes as f64 / args.catalog_size as f64)),
                    ]);
                }
                println!("{}", tbl);
            }
        }
    }

    // --- Memory usage ---
    println!("\nMemory (WiredTiger cache):");
    if let Ok(status) = db.run_command(doc! { "serverStatus": 1, "wiredTiger": 1 }).await {
        if let Ok(wt) = status.get_document("wiredTiger") {
            if let Ok(cache) = wt.get_document("cache") {
                let cache_size = bson_to_i64(cache.get("maximum bytes configured"));
                let used = bson_to_i64(cache.get("bytes currently in the cache"));
                let dirty = bson_to_i64(cache.get("tracked dirty bytes in the cache"));
                let read_into = bson_to_i64(cache.get("bytes read into cache"));
                let written = bson_to_i64(cache.get("bytes written from cache"));
                println!("  Cache configured: {}", fmt_bytes(cache_size));
                if cache_size > 0 {
                    println!("  Cache used:       {} ({:.1}%)", fmt_bytes(used), used as f64 / cache_size as f64 * 100.0);
                }
                println!("  Cache dirty:      {}", fmt_bytes(dirty));
                println!("  Bytes read in:    {}", fmt_bytes(read_into));
                println!("  Bytes written:    {}", fmt_bytes(written));
            }
        }
    }

    // Per-index memory: use $indexStats to get access counts
    println!("\nIndex access stats:");
    if let Ok(mut cursor) = coll.aggregate(vec![doc! { "$indexStats": {} }]).await {
        let mut idx_tbl = Table::new();
        idx_tbl.set_header(vec!["Index", "Accesses", "Since"]);
        while let Some(Ok(stat)) = cursor.next().await {
            let name = stat.get_str("name").unwrap_or("?");
            if let Ok(acc) = stat.get_document("accesses") {
                let ops = bson_to_i64(acc.get("ops"));
                let since = acc.get_datetime("since")
                    .map(|d| d.to_string())
                    .unwrap_or_else(|_| "?".to_string());
                idx_tbl.add_row(vec![
                    Cell::new(name),
                    Cell::new(ops),
                    Cell::new(since),
                ]);
            }
        }
        println!("{}", idx_tbl);
    }

    // --- Benchmark ---
    // NOTE: Fresh random positions are generated for EACH method at each radius
    // to avoid MongoDB cache warming from one method benefiting the next.
    let radii = vec![2.0, 5.0, 10.0, 30.0, 60.0, 300.0];

    /// Generate fresh random sky positions (uniform on sphere).
    fn random_positions(rng: &mut impl Rng, n: usize) -> Vec<(f64, f64)> {
        (0..n).map(|_| {
            let ra: f64 = rng.random::<f64>() * 360.0;
            let u: f64 = rng.random::<f64>();
            let dec: f64 = (2.0 * u - 1.0_f64).asin().to_degrees();
            (ra, dec)
        }).collect()
    }

    // Collect structured results for JSON output
    #[derive(serde::Serialize)]
    struct QueryResult {
        radius_arcsec: f64,
        method: String,
        time_secs: f64,
        raw_hits: usize,
        exact_hits: usize,
    }
    let mut json_queries: Vec<QueryResult> = Vec::new();

    // Capture index sizes for JSON
    let mut idx_size_2ds: i64 = 0;
    let mut idx_size_hpx29: i64 = 0;
    let mut idx_size_hpx16: i64 = 0;
    {
        let stats2: Vec<_> = coll.aggregate(vec![doc! { "$collStats": { "storageStats": {} } }]).await?.collect::<Vec<_>>().await;
        if let Some(Ok(sd)) = stats2.first() {
            if let Ok(s) = sd.get_document("storageStats") {
                if let Ok(isz) = s.get_document("indexSizes") {
                    idx_size_2ds = bson_to_i64(isz.get("loc_2dsphere"));
                    idx_size_hpx29 = bson_to_i64(isz.get("hpx_1"));
                    idx_size_hpx16 = bson_to_i64(isz.get(&format!("hpx_{}_1", args.in_depth)));
                }
            }
        }
    }

    let mut tbl = Table::new();
    tbl.set_header(vec![
        "Radius", "Method", "Depth", "Pixels/Ranges",
        "Time (s)", "Raw hits", "Exact hits", "Overincl.",
    ]);

    println!("\n=== Query benchmark: {} queries/radius ===\n", args.n_queries);

    for radius_arcsec in &radii {
        let radius_rad = (*radius_arcsec / 3600.0_f64).to_radians();
        let range_depth = auto_query_depth(radius_rad);

        // Explain (uses a single sample position)
        if args.explain {
            let sample = random_positions(&mut rng, 1);
            println!("--- {}\" ---", radius_arcsec);
            let geo_f = geojson_filter(sample[0].0, sample[0].1, radius_rad);
            explain_query(&db, coll_name, &geo_f, "2dsphere").await?;
            let (range_f, _) = hpx_range_filter(sample[0].0, sample[0].1, radius_rad, range_depth);
            explain_query(&db, coll_name, &range_f, "HPX ranges").await?;
            let (in_f, _) = hpx_in_filter(sample[0].0, sample[0].1, radius_rad, args.in_depth);
            explain_query(&db, coll_name, &in_f, "HPX $in").await?;
            println!();
        }

        // Fresh random positions per method to avoid cache warming bias.
        // Each method gets its own set so the first method can't warm
        // MongoDB's WiredTiger page cache for the second.
        let pos_geo = random_positions(&mut rng, args.n_queries);
        let pos_ranges = random_positions(&mut rng, args.n_queries);
        let pos_in = random_positions(&mut rng, args.n_queries);

        // 2dsphere
        let geo = bench_geojson(&coll, &pos_geo, *radius_arcsec).await?;

        // HEALPix ranges
        let (ranges, n_ranges) = bench_hpx_ranges(&coll, &pos_ranges, *radius_arcsec, range_depth).await?;

        // HEALPix $in
        let (in_result, n_pixels) = bench_hpx_in(&coll, &pos_in, *radius_arcsec, args.in_depth).await?;

        let radius_label = format!("{}\"", radius_arcsec);

        // 2dsphere row
        tbl.add_row(vec![
            Cell::new(&radius_label),
            Cell::new("2dsphere"),
            Cell::new("-"),
            Cell::new("-"),
            Cell::new(format!("{:.3}", geo.total_time.as_secs_f64())),
            Cell::new(geo.raw_hits),
            Cell::new(geo.exact_hits),
            Cell::new("-"),
        ]);

        // Ranges row
        let range_overincl = if ranges.exact_hits > 0 {
            format!("{:.0}%", (ranges.raw_hits as f64 / ranges.exact_hits as f64 - 1.0) * 100.0)
        } else if ranges.raw_hits > 0 {
            format!("+{}", ranges.raw_hits)
        } else { "0%".into() };

        tbl.add_row(vec![
            Cell::new(""),
            Cell::new("HPX ranges"),
            Cell::new(range_depth),
            Cell::new(format!("{} ranges", n_ranges)),
            Cell::new(format!("{:.3}", ranges.total_time.as_secs_f64())),
            Cell::new(ranges.raw_hits),
            Cell::new(ranges.exact_hits),
            Cell::new(&range_overincl),
        ]);

        // $in row
        let in_overincl = if in_result.exact_hits > 0 {
            format!("{:.0}%", (in_result.raw_hits as f64 / in_result.exact_hits as f64 - 1.0) * 100.0)
        } else if in_result.raw_hits > 0 {
            format!("+{}", in_result.raw_hits)
        } else { "0%".into() };

        tbl.add_row(vec![
            Cell::new(""),
            Cell::new(format!("HPX $in")),
            Cell::new(args.in_depth),
            Cell::new(format!("{} pixels", n_pixels)),
            Cell::new(format!("{:.3}", in_result.total_time.as_secs_f64())),
            Cell::new(in_result.raw_hits),
            Cell::new(in_result.exact_hits),
            Cell::new(&in_overincl),
        ]);

        // Record for JSON
        json_queries.push(QueryResult { radius_arcsec: *radius_arcsec, method: "2dsphere".into(), time_secs: geo.total_time.as_secs_f64(), raw_hits: geo.raw_hits, exact_hits: geo.exact_hits });
        json_queries.push(QueryResult { radius_arcsec: *radius_arcsec, method: "hpx_ranges".into(), time_secs: ranges.total_time.as_secs_f64(), raw_hits: ranges.raw_hits, exact_hits: ranges.exact_hits });
        json_queries.push(QueryResult { radius_arcsec: *radius_arcsec, method: "hpx_in".into(), time_secs: in_result.total_time.as_secs_f64(), raw_hits: in_result.raw_hits, exact_hits: in_result.exact_hits });

        // Note: exact hits differ across methods because each uses fresh random
        // positions (to avoid cache warming bias). This is expected behavior.
    }

    println!("{}", tbl);
    println!("\nOverincl. = (raw_hits / exact_hits - 1) × 100%. Post-filtered to exact distance.");
    println!("Exact hits should match across all methods (if not, indicates a bug).");

    // --- Post-query memory snapshot ---
    println!("\nMemory after queries:");
    if let Ok(status) = db.run_command(doc! { "serverStatus": 1, "wiredTiger": 1 }).await {
        if let Ok(wt) = status.get_document("wiredTiger") {
            if let Ok(cache) = wt.get_document("cache") {
                let cache_size = bson_to_i64(cache.get("maximum bytes configured"));
                let used = bson_to_i64(cache.get("bytes currently in the cache"));
                let read_into = bson_to_i64(cache.get("bytes read into cache"));
                if cache_size > 0 {
                    println!("  Cache used: {} / {} ({:.1}%)",
                        fmt_bytes(used), fmt_bytes(cache_size),
                        used as f64 / cache_size as f64 * 100.0);
                }
                println!("  Total read into cache: {}", fmt_bytes(read_into));
            }
        }
    }

    // Index access counts after benchmark
    println!("\nIndex accesses (total across all queries):");
    if let Ok(mut cursor) = coll.aggregate(vec![doc! { "$indexStats": {} }]).await {
        let mut idx_tbl = Table::new();
        idx_tbl.set_header(vec!["Index", "Accesses"]);
        while let Some(Ok(stat)) = cursor.next().await {
            let name = stat.get_str("name").unwrap_or("?");
            if let Ok(acc) = stat.get_document("accesses") {
                let ops = bson_to_i64(acc.get("ops"));
                idx_tbl.add_row(vec![Cell::new(name), Cell::new(ops)]);
            }
        }
        println!("{}", idx_tbl);
    }

    // --- Write JSON ---
    if let Some(ref path) = args.output_json {
        #[derive(serde::Serialize)]
        struct BenchOutput {
            catalog_size: usize,
            n_queries: usize,
            index_build_secs: std::collections::HashMap<String, f64>,
            index_size_bytes: std::collections::HashMap<String, i64>,
            queries: Vec<QueryResult>,
        }
        let mut build_times = std::collections::HashMap::new();
        build_times.insert("2dsphere".into(), geo_t.as_secs_f64());
        build_times.insert("hpx_29".into(), hpx_t.as_secs_f64());
        build_times.insert(format!("hpx_{}", args.in_depth), in_t.as_secs_f64());

        let mut sizes = std::collections::HashMap::new();
        sizes.insert("2dsphere".into(), idx_size_2ds);
        sizes.insert("hpx_29".into(), idx_size_hpx29);
        sizes.insert(format!("hpx_{}", args.in_depth), idx_size_hpx16);

        let output = BenchOutput {
            catalog_size: args.catalog_size,
            n_queries: args.n_queries,
            index_build_secs: build_times,
            index_size_bytes: sizes,
            queries: json_queries,
        };
        let json = serde_json::to_string_pretty(&output)?;
        std::fs::write(path, &json)?;
        println!("\nResults written to {}", path);
    }

    if !args.keep {
        coll.drop().await?;
        println!("\nCollection dropped. Use --keep to retain.");
    }

    Ok(())
}
