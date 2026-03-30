const std = @import("std");
const testing = std.testing;

/// Compute shingles (k-grams) from a token sequence.
/// Each shingle is hashed to a u64 using FNV-1a.
/// Returns the number of shingles written to `out`.
export fn compute_shingles(
    tokens: [*]const u32,
    token_count: usize,
    k: usize,
    out: [*]u64,
) usize {
    if (token_count < k or k == 0) return 0;

    const count = token_count - k + 1;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        var hash: u64 = 14695981039346656037; // FNV offset basis
        var j: usize = 0;
        while (j < k) : (j += 1) {
            const token = tokens[i + j];
            const bytes = std.mem.asBytes(&token);
            for (bytes) |b| {
                hash ^= @as(u64, b);
                hash *%= 1099511628211; // FNV prime
            }
        }
        out[i] = hash;
    }
    return count;
}

/// Compute shingles for multiple snippets in batch.
/// `snippet_offsets` marks where each snippet starts in `tokens`.
/// `snippet_count` is the number of snippets.
/// `k` is the shingle size.
/// `out_offsets` is filled with offsets into `out_hashes` where each snippet's shingles start.
/// `out_hashes` receives all shingle hashes concatenated.
/// `out_hashes_cap` is the capacity of `out_hashes`.
/// Returns total number of shingles written, or 0 if capacity exceeded.
export fn compute_shingles_batch(
    tokens: [*]const u32,
    token_counts: [*]const usize,
    snippet_count: usize,
    k: usize,
    out_hashes: [*]u64,
    out_hashes_cap: usize,
    out_offsets: [*]usize,
) usize {
    var offset: usize = 0;
    var token_offset: usize = 0;

    var i: usize = 0;
    while (i < snippet_count) : (i += 1) {
        const tc = token_counts[i];
        out_offsets[i] = offset;

        if (tc >= k and k > 0) {
            const count = tc - k + 1;
            if (offset + count > out_hashes_cap) return 0;

            var j: usize = 0;
            while (j < count) : (j += 1) {
                var hash: u64 = 14695981039346656037;
                var m: usize = 0;
                while (m < k) : (m += 1) {
                    const token = tokens[token_offset + j + m];
                    const bytes = std.mem.asBytes(&token);
                    for (bytes) |b| {
                        hash ^= @as(u64, b);
                        hash *%= 1099511628211;
                    }
                }
                out_hashes[offset + j] = hash;
            }
            offset += count;
        }

        token_offset += tc;
    }
    return offset;
}

test "compute_shingles basic" {
    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    var out: [3]u64 = undefined;
    const count = compute_shingles(&tokens, 5, 3, &out);
    try testing.expectEqual(@as(usize, 3), count);

    // Same token sequences should produce same hashes
    const tokens2 = [_]u32{ 1, 2, 3, 4, 5 };
    var out2: [3]u64 = undefined;
    const count2 = compute_shingles(&tokens2, 5, 3, &out2);
    try testing.expectEqual(count, count2);
    for (0..count) |idx| {
        try testing.expectEqual(out[idx], out2[idx]);
    }
}

test "compute_shingles empty input" {
    const tokens = [_]u32{};
    var out: [1]u64 = undefined;
    const count = compute_shingles(&tokens, 0, 3, &out);
    try testing.expectEqual(@as(usize, 0), count);
}

test "compute_shingles k larger than sequence" {
    const tokens = [_]u32{ 1, 2 };
    var out: [1]u64 = undefined;
    const count = compute_shingles(&tokens, 2, 5, &out);
    try testing.expectEqual(@as(usize, 0), count);
}

test "compute_shingles_batch basic" {
    // Two snippets: [1,2,3] and [4,5,6,7]
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const counts = [_]usize{ 3, 4 };
    var hashes: [10]u64 = undefined;
    var offsets: [2]usize = undefined;
    const total = compute_shingles_batch(&tokens, &counts, 2, 2, &hashes, 10, &offsets);
    // snippet 1: 3-2+1=2 shingles, snippet 2: 4-2+1=3 shingles => total 5
    try testing.expectEqual(@as(usize, 5), total);
    try testing.expectEqual(@as(usize, 0), offsets[0]);
    try testing.expectEqual(@as(usize, 2), offsets[1]);
}
