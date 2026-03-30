const std = @import("std");
const testing = std.testing;

/// AST node type as a compact enum
const NodeType = enum(u8) {
    keyword = 0,
    identifier = 1,
    operator = 2,
    literal = 3,
    modifier = 4,
    type_name = 5,
    separator = 6,
    declaration = 7,
    expression = 8,
    statement = 9,
    other = 255,
};

/// Compute AST statistics: node count, max depth, and node type histogram.
/// `nodes` is an array of node type IDs.
/// `depths` is an array of depths for each node.
/// `parent_indices` is an array of parent indices (-1 for root, encoded as maxInt(usize)).
export fn compute_ast_stats(
    nodes: [*]const u8,
    depths: [*]const u16,
    node_count: usize,
    out_max_depth: [*]u16,
    out_histogram: [*]u32,
    histogram_size: usize,
) void {
    // Zero the histogram
    var i: usize = 0;
    while (i < histogram_size) : (i += 1) {
        out_histogram[i] = 0;
    }

    if (node_count == 0) {
        out_max_depth[0] = 0;
        return;
    }

    var max_depth: u16 = 0;

    i = 0;
    while (i < node_count) : (i += 1) {
        const d = depths[i];
        if (d > max_depth) max_depth = d;

        const ntype = nodes[i];
        if (@as(usize, @intCast(ntype)) < histogram_size) {
            out_histogram[@as(usize, @intCast(ntype))] += 1;
        }
    }

    out_max_depth[0] = max_depth;
}

/// Compute parent-child bigram hashes from parent index arrays.
/// Each bigram is (parent_node_type, child_node_type) hashed to u64.
/// `parent_indices` uses maxInt(usize) for root nodes (no parent).
/// Returns the number of bigrams written.
export fn compute_bigrams(
    nodes: [*]const u8,
    parent_indices: [*]const usize,
    node_count: usize,
    out_bigrams: [*]u64,
) usize {
    var count: usize = 0;
    const no_parent = std.math.maxInt(usize);

    var i: usize = 0;
    while (i < node_count) : (i += 1) {
        const parent_idx = parent_indices[i];
        if (parent_idx == no_parent) continue;

        const parent_type = nodes[parent_idx];
        const child_type = nodes[i];

        // Hash: combine parent and child type
        var hash: u64 = 14695981039346656037;
        hash ^= @as(u64, parent_type);
        hash *%= 1099511628211;
        hash ^= @as(u64, child_type);
        hash *%= 1099511628211;

        out_bigrams[count] = hash;
        count += 1;
    }
    return count;
}

/// Compute cyclomatic complexity from control flow node counts.
/// Counts decision points: if, for, while, do, case, catch, &&, ||, ?, throw
export fn compute_cyclomatic_complexity(
    node_types: [*]const u8,
    node_count: usize,
) u32 {
    var cc: u32 = 1; // base complexity

    var i: usize = 0;
    while (i < node_count) : (i += 1) {
        const ntype = node_types[i];
        // NodeType.keyword=0: control flow keywords add to complexity
        // NodeType.operator=2: &&, ||, ? add to complexity
        if (ntype == 0 or ntype == 2) {
            cc += 1;
        }
    }
    return cc;
}

/// Cosine similarity between two histograms (u32 arrays).
/// Returns fixed-point value * 10000.
export fn histogram_cosine_similarity(
    hist_a: [*]const u32,
    hist_b: [*]const u32,
    size: usize,
) u32 {
    if (size == 0) return 0;

    var dot: u64 = 0;
    var norm_a_sq: u64 = 0;
    var norm_b_sq: u64 = 0;

    var i: usize = 0;
    while (i < size) : (i += 1) {
        const a = @as(u64, hist_a[i]);
        const b = @as(u64, hist_b[i]);
        dot += a * b;
        norm_a_sq += a * a;
        norm_b_sq += b * b;
    }

    if (norm_a_sq == 0 or norm_b_sq == 0) return 0;

    // cos_sim = dot / (sqrt(norm_a) * sqrt(norm_b))
    // We compute dot^2 / (norm_a * norm_b) and then sqrt, scaled by 10000
    const num = dot * dot * 10000 * 10000;
    const den = norm_a_sq * norm_b_sq;
    if (den == 0) return 0;

    const result = num / den;
    // Integer sqrt
    return @as(u32, @intCast(isqrt(result)));
}

/// Integer square root using Newton's method
fn isqrt(n: u64) u64 {
    if (n == 0) return 0;
    var x = n;
    var y = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    return x;
}

test "compute_ast_stats basic" {
    // 5 nodes: keyword, identifier, operator, literal, identifier
    const nodes = [_]u8{ 0, 1, 2, 3, 1 };
    const depths = [_]u16{ 0, 1, 1, 2, 2 };
    var max_depth: [1]u16 = undefined;
    var histogram = [_]u32{0} ** 10;

    compute_ast_stats(&nodes, &depths, 5, &max_depth, &histogram, 10);

    try testing.expectEqual(@as(u16, 2), max_depth[0]);
    try testing.expectEqual(@as(u32, 1), histogram[0]); // 1 keyword
    try testing.expectEqual(@as(u32, 2), histogram[1]); // 2 identifiers
    try testing.expectEqual(@as(u32, 1), histogram[2]); // 1 operator
    try testing.expectEqual(@as(u32, 1), histogram[3]); // 1 literal
}

test "compute_ast_stats empty" {
    const nodes = [_]u8{};
    const depths = [_]u16{};
    var max_depth: [1]u16 = undefined;
    var histogram = [_]u32{0} ** 10;

    compute_ast_stats(&nodes, &depths, 0, &max_depth, &histogram, 10);

    try testing.expectEqual(@as(u16, 0), max_depth[0]);
}

test "histogram_cosine_similarity identical" {
    const hist = [_]u32{ 1, 2, 3, 0, 1 };
    const sim = histogram_cosine_similarity(&hist, &hist, 5);
    try testing.expect(sim > 9900); // ~1.0 * 10000
}

test "histogram_cosine_similarity orthogonal" {
    const hist_a = [_]u32{ 1, 0, 0, 0, 0 };
    const hist_b = [_]u32{ 0, 1, 0, 0, 0 };
    const sim = histogram_cosine_similarity(&hist_a, &hist_b, 5);
    try testing.expectEqual(@as(u32, 0), sim);
}

test "compute_bigrams basic" {
    // Root (node 0) has children 1 and 2; node 1 has child 3
    const nodes = [_]u8{ 0, 1, 2, 3 };
    const parents = [_]usize{ std.math.maxInt(usize), 0, 0, 1 };
    var bigrams: [4]u64 = undefined;
    const count = compute_bigrams(&nodes, &parents, 4, &bigrams);
    try testing.expectEqual(@as(usize, 3), count);
}

test "compute_cyclomatic_complexity" {
    // 5 nodes with mixed types (keywords and operators add complexity)
    const nodes = [_]u8{ 0, 1, 0, 2, 1 }; // 2 keywords + 1 operator = 1 + 3 = 4
    const cc = compute_cyclomatic_complexity(&nodes, 5);
    try testing.expectEqual(@as(u32, 4), cc);
}
