const std = @import("std");
const testing = std.testing;

/// Generate a MinHash signature from a set of shingle hashes.
/// Uses `num_hashes` hash functions based on universal hashing: h(x) = (a*x + b) mod p
/// where a and b are random coefficients derived from a seed.
export fn minhash_signature(
    shingles: [*]const u64,
    shingle_count: usize,
    num_hashes: usize,
    seed: u64,
    out_signature: [*]u64,
) void {
    const p: u64 = 0xFFFFFFFFFFFFFFC5; // large prime

    // Initialize all signatures to max
    var i: usize = 0;
    while (i < num_hashes) : (i += 1) {
        out_signature[i] = std.math.maxInt(u64);
    }

    // For each hash function, compute h(x) = (a*x + b) mod p for each shingle
    i = 0;
    while (i < num_hashes) : (i += 1) {
        const a = hash_mix(seed, i * 2 + 1) | 1; // ensure a is odd
        const b = hash_mix(seed, i * 2 + 2);

        var j: usize = 0;
        while (j < shingle_count) : (j += 1) {
            const val = a *% shingles[j] +% b;
            const h = val % p;
            if (h < out_signature[i]) {
                out_signature[i] = h;
            }
        }
    }
}

/// Compute Jaccard similarity estimate from two MinHash signatures.
/// Returns a fixed-point value * 10000 (e.g., 5000 = 0.5 similarity).
export fn minhash_similarity(
    sig_a: [*]const u64,
    sig_b: [*]const u64,
    num_hashes: usize,
) u32 {
    if (num_hashes == 0) return 0;

    var matches: usize = 0;
    var i: usize = 0;
    while (i < num_hashes) : (i += 1) {
        if (sig_a[i] == sig_b[i]) matches += 1;
    }

    return @as(u32, @intCast((matches * 10000) / num_hashes));
}

/// Hash-based LSH: divide signature into bands, hash each band to a bucket.
/// Returns bucket IDs for each band.
export fn lsh_buckets(
    signature: [*]const u64,
    sig_len: usize,
    bands: usize,
    rows_per_band: usize,
    out_buckets: [*]u64,
) void {
    var band: usize = 0;
    while (band < bands) : (band += 1) {
        const start = band * rows_per_band;
        if (start + rows_per_band > sig_len) {
            out_buckets[band] = 0;
            continue;
        }

        var hash: u64 = 14695981039346656037; // FNV offset
        var r: usize = 0;
        while (r < rows_per_band) : (r += 1) {
            const val = signature[start + r];
            const bytes = std.mem.asBytes(&val);
            for (bytes) |b| {
                hash ^= @as(u64, b);
                hash *%= 1099511628211;
            }
        }
        out_buckets[band] = hash;
    }
}

/// Simple mixing function for generating hash coefficients
fn hash_mix(seed: u64, counter: u64) u64 {
    var h = seed ^ (counter *% 0x9E3779B97F4A7C15);
    h ^= h >> 30;
    h *%= 0xBF58476D1CE4E5B9;
    h ^= h >> 27;
    h *%= 0x94D049BB133111EB;
    h ^= h >> 31;
    return h;
}

test "minhash_signature basic" {
    const shingles = [_]u64{ 100, 200, 300, 400, 500 };
    var sig: [10]u64 = undefined;
    minhash_signature(&shingles, 5, 10, 42, &sig);

    // All signatures should be populated
    for (0..10) |i| {
        try testing.expect(sig[i] < std.math.maxInt(u64));
    }
}

test "minhash_similarity identical" {
    const shingles = [_]u64{ 100, 200, 300, 400, 500 };
    var sig_a: [10]u64 = undefined;
    var sig_b: [10]u64 = undefined;
    minhash_signature(&shingles, 5, 10, 42, &sig_a);
    minhash_signature(&shingles, 5, 10, 42, &sig_b);

    const sim = minhash_similarity(&sig_a, &sig_b, 10);
    try testing.expectEqual(@as(u32, 10000), sim); // identical = 1.0
}

test "minhash_similarity different" {
    const shingles_a = [_]u64{ 100, 200, 300 };
    const shingles_b = [_]u64{ 999, 888, 777 };
    var sig_a: [20]u64 = undefined;
    var sig_b: [20]u64 = undefined;
    minhash_signature(&shingles_a, 3, 20, 42, &sig_a);
    minhash_signature(&shingles_b, 3, 20, 42, &sig_b);

    const sim = minhash_similarity(&sig_a, &sig_b, 20);
    // Different inputs should generally have low similarity
    try testing.expect(sim < 5000);
}

test "lsh_buckets basic" {
    const shingles = [_]u64{ 100, 200, 300, 400, 500, 600, 700, 800 };
    var sig: [8]u64 = undefined;
    minhash_signature(&shingles, 8, 8, 42, &sig);

    var buckets: [4]u64 = undefined;
    lsh_buckets(&sig, 8, 4, 2, &buckets);

    // All buckets should be populated
    for (0..4) |i| {
        try testing.expect(buckets[i] != 0);
    }

    // Same signature should produce same buckets
    var buckets2: [4]u64 = undefined;
    lsh_buckets(&sig, 8, 4, 2, &buckets2);
    for (0..4) |i| {
        try testing.expectEqual(buckets[i], buckets2[i]);
    }
}
