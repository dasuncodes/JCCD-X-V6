const std = @import("std");
const testing = std.testing;

/// Remove single-line (//) and multi-line (/* */) comments from Java source.
/// Respects string literals and char literals.
/// Returns the number of bytes written to `out`.
export fn remove_comments(
    source: [*]const u8,
    src_len: usize,
    out: [*]u8,
    out_cap: usize,
) usize {
    var i: usize = 0;
    var o: usize = 0;
    var in_string = false;
    var in_char = false;
    var in_line_comment = false;
    var in_block_comment = false;

    while (i < src_len) {
        if (o >= out_cap) break;
        const c = source[i];

        if (in_line_comment) {
            if (c == '\n') {
                in_line_comment = false;
                out[o] = '\n';
                o += 1;
            }
            i += 1;
            continue;
        }

        if (in_block_comment) {
            if (c == '*' and i + 1 < src_len and source[i + 1] == '/') {
                in_block_comment = false;
                i += 2;
            } else {
                if (c == '\n') {
                    if (o < out_cap) {
                        out[o] = '\n';
                        o += 1;
                    }
                }
                i += 1;
            }
            continue;
        }

        if (in_string) {
            out[o] = c;
            o += 1;
            if (c == '\\' and i + 1 < src_len) {
                i += 1;
                if (o < out_cap) {
                    out[o] = source[i];
                    o += 1;
                }
            } else if (c == '"') {
                in_string = false;
            }
            i += 1;
            continue;
        }

        if (in_char) {
            out[o] = c;
            o += 1;
            if (c == '\\' and i + 1 < src_len) {
                i += 1;
                if (o < out_cap) {
                    out[o] = source[i];
                    o += 1;
                }
            } else if (c == '\'') {
                in_char = false;
            }
            i += 1;
            continue;
        }

        // Check for comment starts
        if (c == '/' and i + 1 < src_len) {
            if (source[i + 1] == '/') {
                in_line_comment = true;
                i += 2;
                continue;
            }
            if (source[i + 1] == '*') {
                in_block_comment = true;
                i += 2;
                continue;
            }
        }

        if (c == '"') {
            in_string = true;
        }
        if (c == '\'') {
            in_char = true;
        }

        out[o] = c;
        o += 1;
        i += 1;
    }
    return o;
}

/// Normalize whitespace: collapse runs of spaces/tabs, remove blank lines.
/// Preserves newlines that separate code lines.
/// Returns bytes written to `out`.
export fn normalize_whitespace(
    source: [*]const u8,
    src_len: usize,
    out: [*]u8,
    out_cap: usize,
) usize {
    var i: usize = 0;
    var o: usize = 0;
    var last_was_space = false;
    var last_was_newline = true; // start as if after newline to skip leading blank lines
    var line_has_content = false;

    while (i < src_len) {
        if (o >= out_cap) break;
        const c = source[i];

        if (c == '\n') {
            if (line_has_content) {
                out[o] = '\n';
                o += 1;
            }
            last_was_space = false;
            last_was_newline = true;
            line_has_content = false;
            i += 1;
            continue;
        }

        if (c == ' ' or c == '\t' or c == '\r') {
            if (line_has_content and !last_was_space) {
                if (o < out_cap) {
                    out[o] = ' ';
                    o += 1;
                }
            }
            last_was_space = true;
            i += 1;
            continue;
        }

        line_has_content = true;
        last_was_space = false;
        last_was_newline = false;
        out[o] = c;
        o += 1;
        i += 1;
    }

    // Remove trailing newline
    if (o > 0 and out[o - 1] == '\n') {
        o -= 1;
    }
    return o;
}

/// Full normalization pipeline: comments + whitespace.
/// Returns bytes written to `out`.
export fn normalize_source(
    source: [*]const u8,
    src_len: usize,
    out: [*]u8,
    out_cap: usize,
) usize {
    // Two-pass: first remove comments, then normalize whitespace.
    // Use a temporary buffer (on stack if small enough).
    var tmp_buf: [65536]u8 = undefined;
    const use_heap = src_len > tmp_buf.len;

    if (use_heap) {
        // For large inputs, use the output buffer as intermediate
        // and do both passes in-place with careful ordering.
        const comment_len = remove_comments(source, src_len, out, out_cap);
        // Now normalize whitespace in-place (reading from out, writing to tmp then back)
        // Actually, we can read and write from the same buffer if we're careful
        // since normalization only shrinks.
        var i: usize = 0;
        var o: usize = 0;
        var last_was_space = false;
        var line_has_content = false;

        while (i < comment_len) {
            if (o >= out_cap) break;
            const c = out[i];
            if (c == '\n') {
                if (line_has_content) {
                    out[o] = '\n';
                    o += 1;
                }
                last_was_space = false;
                line_has_content = false;
                i += 1;
                continue;
            }
            if (c == ' ' or c == '\t' or c == '\r') {
                if (line_has_content and !last_was_space) {
                    if (o < out_cap) {
                        out[o] = ' ';
                        o += 1;
                    }
                }
                last_was_space = true;
                i += 1;
                continue;
            }
            line_has_content = true;
            last_was_space = false;
            out[o] = c;
            o += 1;
            i += 1;
        }
        if (o > 0 and out[o - 1] == '\n') o -= 1;
        return o;
    } else {
        const comment_len = remove_comments(source, src_len, &tmp_buf, tmp_buf.len);
        return normalize_whitespace(&tmp_buf, comment_len, out, out_cap);
    }
}

/// Compute normalization impact metrics.
/// Writes 4 u32 values to out_metrics:
/// [0] = original line count
/// [1] = normalized line count
/// [2] = blank lines removed
/// [3] = comments removed (lines that were entirely comments)
export fn compute_norm_impact(
    original: [*]const u8,
    orig_len: usize,
    normalized: [*]const u8,
    norm_len: usize,
    out_metrics: [*]u32,
) void {
    out_metrics[0] = count_lines(original, orig_len);
    out_metrics[1] = count_lines(normalized, norm_len);
    out_metrics[2] = if (out_metrics[0] > out_metrics[1]) out_metrics[0] - out_metrics[1] else 0;
    out_metrics[3] = 0; // placeholder; detailed comment counting done in Python
}

fn count_lines(buf: [*]const u8, len: usize) u32 {
    if (len == 0) return 0;
    var count: u32 = 1;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        if (buf[i] == '\n') count += 1;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "remove_comments single line" {
    const src = "int x = 5; // this is a comment\nint y = 10;";
    var out: [256]u8 = undefined;
    const len = remove_comments(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expect(std.mem.indexOf(u8, result, "//") == null);
    try testing.expect(std.mem.indexOf(u8, result, "int x = 5;") != null);
    try testing.expect(std.mem.indexOf(u8, result, "int y = 10;") != null);
}

test "remove_comments multi line" {
    const src = "int x = 5; /* block\ncomment */ int y = 10;";
    var out: [256]u8 = undefined;
    const len = remove_comments(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expect(std.mem.indexOf(u8, result, "/*") == null);
    try testing.expect(std.mem.indexOf(u8, result, "int x = 5;") != null);
    try testing.expect(std.mem.indexOf(u8, result, "int y = 10;") != null);
}

test "remove_comments preserves strings" {
    const src = "String s = \"// not a comment\"; // real comment";
    var out: [256]u8 = undefined;
    const len = remove_comments(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expect(std.mem.indexOf(u8, result, "\"// not a comment\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "// real comment") == null);
}

test "remove_comments preserves char literals" {
    const src = "char c = '/'; // comment";
    var out: [256]u8 = undefined;
    const len = remove_comments(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expect(std.mem.indexOf(u8, result, "'/'") != null);
}

test "normalize_whitespace collapses spaces" {
    const src = "int    x   =    5;";
    var out: [256]u8 = undefined;
    const len = normalize_whitespace(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expectEqualStrings("int x = 5;", result);
}

test "normalize_whitespace removes blank lines" {
    const src = "int x = 5;\n\n\n\nint y = 10;";
    var out: [256]u8 = undefined;
    const len = normalize_whitespace(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expectEqualStrings("int x = 5;\nint y = 10;", result);
}

test "normalize_whitespace handles tabs" {
    const src = "\tint\tx\t=\t5;";
    var out: [256]u8 = undefined;
    const len = normalize_whitespace(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expectEqualStrings("int x = 5;", result);
}

test "normalize_source full pipeline" {
    const src = "// header comment\nint x = 5;  /* block\n   comment */  int y = 10;\n\n\n";
    var out: [256]u8 = undefined;
    const len = normalize_source(src, src.len, &out, out.len);
    const result = out[0..len];
    try testing.expect(std.mem.indexOf(u8, result, "//") == null);
    try testing.expect(std.mem.indexOf(u8, result, "/*") == null);
    try testing.expect(std.mem.indexOf(u8, result, "int x = 5;") != null);
    try testing.expect(std.mem.indexOf(u8, result, "int y = 10;") != null);
}

test "normalize_source empty" {
    const src = "";
    var out: [256]u8 = undefined;
    const len = normalize_source(src, src.len, &out, out.len);
    try testing.expectEqual(@as(usize, 0), len);
}

test "compute_norm_impact" {
    const orig = "int x = 5;\n\n\n// comment\nint y = 10;\n";
    const norm = "int x = 5;\nint y = 10;";
    var metrics: [4]u32 = undefined;
    compute_norm_impact(orig, orig.len, norm, norm.len, &metrics);
    try testing.expect(metrics[0] > metrics[1]); // original has more lines
}
