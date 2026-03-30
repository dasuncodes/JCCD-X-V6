const std = @import("std");
const testing = std.testing;
const ThreadPool = std.Thread.Pool;

/// Number of features computed per pair
pub const FEATURE_COUNT: usize = 25;

/// Compute features for a single pair.
/// tokens_a/b: token sequences
/// counts_a/b: token counts
/// hist_a/b: AST histograms (12 u32 values each)
/// depths_a/b: max AST depths
/// nodes_a/b: AST node counts
/// cc_a/b: cyclomatic complexities
/// bigrams_a/b: bigram hash arrays
/// bigram_counts_a/b: number of bigrams
/// out: output array of FEATURE_COUNT f32 values
export fn compute_pair_features(
    tokens_a: [*]const u32,
    count_a: usize,
    tokens_b: [*]const u32,
    count_b: usize,
    hist_a: [*]const u32,
    hist_b: [*]const u32,
    depth_a: u16,
    depth_b: u16,
    node_count_a: u32,
    node_count_b: u32,
    cc_a: u32,
    cc_b: u32,
    bigrams_a: [*]const u64,
    bigram_count_a: usize,
    bigrams_b: [*]const u64,
    bigram_count_b: usize,
    out: [*]f32,
) void {
    // --- Lexical features (5) ---
    // 0: levenshtein placeholder (computed in Python with rapidfuzz)
    out[0] = 0.0;

    // 1: Jaccard on token sets
    out[1] = tokenJaccard(tokens_a, count_a, tokens_b, count_b);

    // 2: Dice on token sets
    out[2] = tokenDice(tokens_a, count_a, tokens_b, count_b);

    // 3: LCS ratio
    out[3] = lcsRatio(tokens_a, count_a, tokens_b, count_b);

    // 4: Sequence match ratio (approximation using LCS)
    out[4] = out[3]; // LCS ratio as approximation

    // --- Structural features (8) ---
    // 5: AST histogram cosine
    out[5] = histCosineF32(hist_a, hist_b, 12);

    // 6: Bigram cosine (Jaccard approximation)
    out[6] = bigramJaccard(bigrams_a, bigram_count_a, bigrams_b, bigram_count_b);

    // 7: depth diff
    out[7] = @as(f32, @floatFromInt(if (depth_a > depth_b) depth_a - depth_b else depth_b - depth_a));

    // 8: depth ratio
    const dmax = @max(depth_a, depth_b);
    out[8] = if (dmax > 0) @as(f32, @floatFromInt(@min(depth_a, depth_b))) / @as(f32, @floatFromInt(dmax)) else 1.0;

    // 9: node count 1
    out[9] = @as(f32, @floatFromInt(node_count_a));

    // 10: node count 2
    out[10] = @as(f32, @floatFromInt(node_count_b));

    // 11: node ratio
    const nmax = @max(node_count_a, node_count_b);
    out[11] = if (nmax > 0) @as(f32, @floatFromInt(@min(node_count_a, node_count_b))) / @as(f32, @floatFromInt(nmax)) else 1.0;

    // 12: node diff
    out[12] = @as(f32, @floatFromInt(if (node_count_a > node_count_b) node_count_a - node_count_b else node_count_b - node_count_a));

    // --- Quantitative features (12) ---
    // 13: token ratio
    const tmax = @max(count_a, count_b);
    out[13] = if (tmax > 0) @as(f32, @floatFromInt(@min(count_a, count_b))) / @as(f32, @floatFromInt(tmax)) else 1.0;

    // 14: identifier ratio (Dice on IDENTIFIER tokens = type 1)
    out[14] = identifierDice(tokens_a, count_a, tokens_b, count_b);

    // 15: cc1
    out[15] = @as(f32, @floatFromInt(cc_a));

    // 16: cc2
    out[16] = @as(f32, @floatFromInt(cc_b));

    // 17: cc ratio
    const ccmax = @max(cc_a, cc_b);
    out[17] = if (ccmax > 0) @as(f32, @floatFromInt(@min(cc_a, cc_b))) / @as(f32, @floatFromInt(ccmax)) else 1.0;

    // 18: cc diff
    out[18] = @as(f32, @floatFromInt(if (cc_a > cc_b) cc_a - cc_b else cc_b - cc_a));
}

/// Batch compute features for all pairs. Parallelized with thread pool.
/// pair_idx_a/b: indices into the token/ast arrays for each pair
/// pair_count: number of pairs
/// tokens: all token sequences concatenated
/// token_offsets: start index in tokens for each file
/// token_counts: token count for each file
/// histograms: 12 u32 values per file, concatenated
/// depths: max depth per file
/// node_counts: node count per file
/// cyclomatics: cyclomatic complexity per file
/// bigrams: all bigram hashes concatenated
/// bigram_offsets: start index in bigrams for each file
/// bigram_counts: bigram count for each file
/// out_features: output, pair_count * FEATURE_COUNT f32 values
/// num_threads: 0 = auto-detect
export fn compute_features_batch(
    pair_idx_a: [*]const usize,
    pair_idx_b: [*]const usize,
    pair_count: usize,
    tokens: [*]const u32,
    token_offsets: [*]const usize,
    token_counts: [*]const usize,
    histograms: [*]const u32,
    depths: [*]const u16,
    node_counts: [*]const u32,
    cyclomatics: [*]const u32,
    bigrams: [*]const u64,
    bigram_offsets: [*]const usize,
    bigram_counts: [*]const usize,
    out_features: [*]f32,
    num_threads: usize,
) void {
    const n_threads = if (num_threads == 0) @max(1, std.Thread.getCpuCount() catch 4) else num_threads;

    // For small batches, don't parallelize
    if (pair_count < 1000) {
        computeFeaturesRange(
            pair_idx_a,
            pair_idx_b,
            0,
            pair_count,
            tokens,
            token_offsets,
            token_counts,
            histograms,
            depths,
            node_counts,
            cyclomatics,
            bigrams,
            bigram_offsets,
            bigram_counts,
            out_features,
        );
        return;
    }

    // Parallel computation
    // Split work into chunks and use a simple thread-per-chunk approach
    const chunk_size = pair_count / n_threads;
    const remainder = pair_count % n_threads;

    var threads: [256]std.Thread = undefined;
    const actual_threads = @min(n_threads, 256);

    var args: [256]ChunkArgs = undefined;
    var offset: usize = 0;
    var t: usize = 0;

    while (t < actual_threads) : (t += 1) {
        const size = chunk_size + if (t < remainder) @as(usize, 1) else 0;
        args[t] = .{
            .pair_idx_a = pair_idx_a,
            .pair_idx_b = pair_idx_b,
            .start = offset,
            .count = size,
            .tokens = tokens,
            .token_offsets = token_offsets,
            .token_counts = token_counts,
            .histograms = histograms,
            .depths = depths,
            .node_counts = node_counts,
            .cyclomatics = cyclomatics,
            .bigrams = bigrams,
            .bigram_offsets = bigram_offsets,
            .bigram_counts = bigram_counts,
            .out_features = out_features,
        };
        threads[t] = std.Thread.spawn(.{}, chunkWorker, .{&args[t]}) catch {
            // If thread creation fails, compute sequentially
            computeFeaturesRange(
                pair_idx_a,
                pair_idx_b,
                offset,
                size,
                tokens,
                token_offsets,
                token_counts,
                histograms,
                depths,
                node_counts,
                cyclomatics,
                bigrams,
                bigram_offsets,
                bigram_counts,
                out_features,
            );
            continue;
        };
        offset += size;
    }

    // Wait for all threads
    t = 0;
    while (t < actual_threads) : (t += 1) {
        threads[t].join();
    }
}

const ChunkArgs = struct {
    pair_idx_a: [*]const usize,
    pair_idx_b: [*]const usize,
    start: usize,
    count: usize,
    tokens: [*]const u32,
    token_offsets: [*]const usize,
    token_counts: [*]const usize,
    histograms: [*]const u32,
    depths: [*]const u16,
    node_counts: [*]const u32,
    cyclomatics: [*]const u32,
    bigrams: [*]const u64,
    bigram_offsets: [*]const usize,
    bigram_counts: [*]const usize,
    out_features: [*]f32,
};

fn chunkWorker(args: *ChunkArgs) void {
    computeFeaturesRange(
        args.pair_idx_a,
        args.pair_idx_b,
        args.start,
        args.count,
        args.tokens,
        args.token_offsets,
        args.token_counts,
        args.histograms,
        args.depths,
        args.node_counts,
        args.cyclomatics,
        args.bigrams,
        args.bigram_offsets,
        args.bigram_counts,
        args.out_features,
    );
}

fn computeFeaturesRange(
    pair_idx_a: [*]const usize,
    pair_idx_b: [*]const usize,
    start: usize,
    count: usize,
    tokens: [*]const u32,
    token_offsets: [*]const usize,
    token_counts: [*]const usize,
    histograms: [*]const u32,
    depths: [*]const u16,
    node_counts: [*]const u32,
    cyclomatics: [*]const u32,
    bigrams: [*]const u64,
    bigram_offsets: [*]const usize,
    bigram_counts: [*]const usize,
    out_features: [*]f32,
) void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const pi = start + i;
        const ia = pair_idx_a[pi];
        const ib = pair_idx_b[pi];

        const tok_a = tokens + token_offsets[ia];
        const cnt_a = token_counts[ia];
        const tok_b = tokens + token_offsets[ib];
        const cnt_b = token_counts[ib];

        const hist_a = histograms + ia * 12;
        const hist_b = histograms + ib * 12;

        const big_a = bigrams + bigram_offsets[ia];
        const bc_a = bigram_counts[ia];
        const big_b = bigrams + bigram_offsets[ib];
        const bc_b = bigram_counts[ib];

        compute_pair_features(
            tok_a,
            cnt_a,
            tok_b,
            cnt_b,
            hist_a,
            hist_b,
            depths[ia],
            depths[ib],
            node_counts[ia],
            node_counts[ib],
            cyclomatics[ia],
            cyclomatics[ib],
            big_a,
            bc_a,
            big_b,
            bc_b,
            out_features + pi * FEATURE_COUNT,
        );
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn tokenJaccard(a: [*]const u32, ca: usize, b: [*]const u32, cb: usize) f32 {
    if (ca == 0 and cb == 0) return 1.0;
    if (ca == 0 or cb == 0) return 0.0;

    // Use a simple approach: count intersection via sorted comparison
    // For efficiency, use hash sets via a stack-allocated bitfield approach
    // Since token IDs are small, we can count unique tokens easily
    var set_a: [8192]u8 = .{0} ** 8192;
    var set_b: [8192]u8 = .{0} ** 8192;

    var i: usize = 0;
    while (i < ca) : (i += 1) {
        const idx = a[i] % 8192;
        set_a[idx] = 1;
    }
    i = 0;
    while (i < cb) : (i += 1) {
        const idx = b[i] % 8192;
        set_b[idx] = 1;
    }

    var intersection: f32 = 0;
    var union_count: f32 = 0;
    i = 0;
    while (i < 8192) : (i += 1) {
        if (set_a[i] == 1 or set_b[i] == 1) {
            union_count += 1;
            if (set_a[i] == 1 and set_b[i] == 1) {
                intersection += 1;
            }
        }
    }

    return if (union_count > 0) intersection / union_count else 0.0;
}

fn tokenDice(a: [*]const u32, ca: usize, b: [*]const u32, cb: usize) f32 {
    if (ca == 0 and cb == 0) return 1.0;
    if (ca == 0 or cb == 0) return 0.0;

    var set_a: [8192]u8 = .{0} ** 8192;
    var set_b: [8192]u8 = .{0} ** 8192;

    var i: usize = 0;
    while (i < ca) : (i += 1) {
        set_a[a[i] % 8192] = 1;
    }
    i = 0;
    while (i < cb) : (i += 1) {
        set_b[b[i] % 8192] = 1;
    }

    var intersection: f32 = 0;
    var size_a: f32 = 0;
    var size_b: f32 = 0;
    i = 0;
    while (i < 8192) : (i += 1) {
        if (set_a[i] == 1) size_a += 1;
        if (set_b[i] == 1) size_b += 1;
        if (set_a[i] == 1 and set_b[i] == 1) intersection += 1;
    }

    return if (size_a + size_b > 0) 2 * intersection / (size_a + size_b) else 0.0;
}

fn lcsRatio(a: [*]const u32, ca: usize, b: [*]const u32, cb: usize) f32 {
    if (ca == 0 and cb == 0) return 1.0;
    if (ca == 0 or cb == 0) return 0.0;

    // Limit to reasonable size to avoid O(n*m) explosion
    const max_len = @min(@max(ca, cb), 512);
    const na = @min(ca, max_len);
    const nb = @min(cb, max_len);

    // Two-row DP
    var prev: [513]u32 = .{0} ** 513;
    var curr: [513]u32 = .{0} ** 513;

    var i: usize = 1;
    while (i <= na) : (i += 1) {
        var j: usize = 1;
        while (j <= nb) : (j += 1) {
            if (a[i - 1] == b[j - 1]) {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = @max(curr[j - 1], prev[j]);
            }
        }
        // Swap
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }

    const lcs_len = @as(f32, @floatFromInt(prev[nb]));
    const total = @as(f32, @floatFromInt(ca + cb));
    return if (total > 0) 2 * lcs_len / total else 0.0;
}

fn histCosineF32(a: [*]const u32, b: [*]const u32, size: usize) f32 {
    var dot: f64 = 0;
    var norm_a: f64 = 0;
    var norm_b: f64 = 0;

    var i: usize = 0;
    while (i < size) : (i += 1) {
        const fa = @as(f64, @floatFromInt(a[i]));
        const fb = @as(f64, @floatFromInt(b[i]));
        dot += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }

    if (norm_a == 0 or norm_b == 0) return 0.0;
    return @as(f32, @floatCast(dot / (@sqrt(norm_a) * @sqrt(norm_b))));
}

fn bigramJaccard(a: [*]const u64, ca: usize, b: [*]const u64, cb: usize) f32 {
    if (ca == 0 and cb == 0) return 0.0;
    if (ca == 0 or cb == 0) return 0.0;

    // Count intersection (naive O(n*m), OK for small bigram sets)
    var intersection: f32 = 0;
    var i: usize = 0;
    while (i < ca) : (i += 1) {
        var j: usize = 0;
        while (j < cb) : (j += 1) {
            if (a[i] == b[j]) {
                intersection += 1;
                break;
            }
        }
    }

    const union_size = @as(f32, @floatFromInt(ca + cb)) - intersection;
    return if (union_size > 0) intersection / union_size else 0.0;
}

fn identifierDice(a: [*]const u32, ca: usize, b: [*]const u32, cb: usize) f32 {
    // IDENTIFIER type = 1
    var count_a: u32 = 0;
    var count_b: u32 = 0;
    var count_both: u32 = 0;

    // For simplicity, just count how many are identifiers
    var i: usize = 0;
    while (i < ca) : (i += 1) {
        if (a[i] == 1) count_a += 1;
    }
    i = 0;
    while (i < cb) : (i += 1) {
        if (b[i] == 1) count_b += 1;
    }

    // Approximate: if both have identifiers, ratio of smaller/larger
    const total = count_a + count_b;
    if (total == 0) return 1.0;
    count_both = @min(count_a, count_b);
    return @as(f32, @floatFromInt(2 * count_both)) / @as(f32, @floatFromInt(total));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "tokenJaccard identical" {
    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const result = tokenJaccard(&tokens, 5, &tokens, 5);
    try testing.expect(result > 0.99);
}

test "tokenJaccard disjoint" {
    const a = [_]u32{ 1, 2, 3 };
    const b = [_]u32{ 100, 200, 300 };
    const result = tokenJaccard(&a, 3, &b, 3);
    try testing.expect(result < 0.01);
}

test "tokenDice identical" {
    const tokens = [_]u32{ 1, 2, 3 };
    const result = tokenDice(&tokens, 3, &tokens, 3);
    try testing.expect(result > 0.99);
}

test "lcsRatio identical" {
    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const result = lcsRatio(&tokens, 5, &tokens, 5);
    try testing.expect(result > 0.99);
}

test "lcsRatio disjoint" {
    const a = [_]u32{ 1, 2, 3 };
    const b = [_]u32{ 4, 5, 6 };
    const result = lcsRatio(&a, 3, &b, 3);
    try testing.expect(result < 0.01);
}

test "histCosineF32 identical" {
    const h = [_]u32{ 1, 2, 3, 0, 1 };
    const result = histCosineF32(&h, &h, 5);
    try testing.expect(result > 0.99);
}

test "histCosineF32 orthogonal" {
    const a = [_]u32{ 1, 0, 0, 0, 0 };
    const b = [_]u32{ 0, 1, 0, 0, 0 };
    const result = histCosineF32(&a, &b, 5);
    try testing.expect(result < 0.01);
}

test "compute_pair_features output count" {
    const tokens = [_]u32{ 1, 2, 3 };
    const hist = [_]u32{ 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const bigrams = [_]u64{ 100, 200 };
    var out: [FEATURE_COUNT]f32 = undefined;
    compute_pair_features(
        &tokens,
        3,
        &tokens,
        3,
        &hist,
        &hist,
        5,
        5,
        10,
        10,
        3,
        3,
        &bigrams,
        2,
        &bigrams,
        2,
        &out,
    );
    // Identical pair should have high similarity for most features
    try testing.expect(out[1] > 0.99); // Jaccard
    try testing.expect(out[2] > 0.99); // Dice
    try testing.expect(out[5] > 0.99); // hist cosine
}
