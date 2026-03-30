const std = @import("std");
const testing = std.testing;

/// Abstract token categories
pub const TokenType = enum(u8) {
    KEYWORD = 0,
    IDENTIFIER = 1,
    OPERATOR = 2,
    LITERAL = 3,
    MODIFIER = 4,
    TYPE_NAME = 5,
    SEPARATOR = 6,
    DECLARATION = 7,
    EXPRESSION = 8,
    STATEMENT = 9,
    ANNOTATION = 10,
    OTHER = 255,
};

/// Token type ID mapping (tree-sitter node type → abstract category).
/// Input: raw node type IDs from tree-sitter.
/// Output: abstract token type IDs.
///
/// The mapping is passed as a lookup table (node_type_to_abstract) of size `vocab_size`.
/// Each entry maps a raw node type index to its abstract TokenType value.
export fn abstract_tokens(
    raw_types: [*]const u16,
    count: usize,
    lookup: [*]const u8,
    vocab_size: usize,
    out_types: [*]u8,
) void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const raw = raw_types[i];
        if (@as(usize, @intCast(raw)) < vocab_size) {
            out_types[i] = lookup[@as(usize, @intCast(raw))];
        } else {
            out_types[i] = 255; // OTHER
        }
    }
}

/// Encode abstract token types to compact integer IDs.
/// Maps u8 token type values to u32 for use with shingling.
export fn encode_tokens(
    token_types: [*]const u8,
    count: usize,
    out_ids: [*]u32,
) void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        out_ids[i] = @as(u32, @intCast(token_types[i]));
    }
}

/// Count tokens by abstract category.
/// Returns counts for each of the 11 categories + OTHER.
export fn count_token_types(
    token_types: [*]const u8,
    count: usize,
    out_counts: [*]u32,
    num_categories: usize,
) void {
    var i: usize = 0;
    while (i < num_categories) : (i += 1) {
        out_counts[i] = 0;
    }
    i = 0;
    while (i < count) : (i += 1) {
        const t = @as(usize, @intCast(token_types[i]));
        if (t < num_categories) {
            out_counts[t] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "abstract_tokens basic" {
    const raw = [_]u16{ 0, 1, 2, 3, 999 };
    // Lookup: 0→KEYWORD(0), 1→IDENTIFIER(1), 2→OPERATOR(2), 3→LITERAL(3)
    const lookup = [_]u8{ 0, 1, 2, 3 };
    var out: [5]u8 = undefined;
    abstract_tokens(&raw, 5, &lookup, 4, &out);
    try testing.expectEqual(@as(u8, 0), out[0]); // KEYWORD
    try testing.expectEqual(@as(u8, 1), out[1]); // IDENTIFIER
    try testing.expectEqual(@as(u8, 2), out[2]); // OPERATOR
    try testing.expectEqual(@as(u8, 3), out[3]); // LITERAL
    try testing.expectEqual(@as(u8, 255), out[4]); // OTHER (out of vocab)
}

test "abstract_tokens empty" {
    const raw = [_]u16{};
    const lookup = [_]u8{0};
    var out: [1]u8 = undefined;
    abstract_tokens(&raw, 0, &lookup, 1, &out);
    // No assertions needed, just ensure no crash
}

test "count_token_types" {
    const types = [_]u8{ 0, 1, 0, 2, 1, 0, 255 };
    var counts: [12]u32 = undefined;
    count_token_types(&types, 7, &counts, 12);
    try testing.expectEqual(@as(u32, 3), counts[0]); // 3 KEYWORDs
    try testing.expectEqual(@as(u32, 2), counts[1]); // 2 IDENTIFIERs
    try testing.expectEqual(@as(u32, 1), counts[2]); // 1 OPERATOR
    try testing.expectEqual(@as(u32, 0), counts[3]); // 0 LITERALs
}

test "encode_tokens" {
    const types = [_]u8{ 0, 1, 2, 3 };
    var ids: [4]u32 = undefined;
    encode_tokens(&types, 4, &ids);
    try testing.expectEqual(@as(u32, 0), ids[0]);
    try testing.expectEqual(@as(u32, 1), ids[1]);
    try testing.expectEqual(@as(u32, 2), ids[2]);
    try testing.expectEqual(@as(u32, 3), ids[3]);
}
