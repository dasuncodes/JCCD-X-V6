const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Shingling library
    const shingling_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/shingling/shingling.zig"),
        .target = target,
        .optimize = optimize,
    });
    const shingling = b.addLibrary(.{
        .name = "shingling",
        .root_module = shingling_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(shingling);

    // LSH / MinHash library
    const minhash_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/lsh/minhash.zig"),
        .target = target,
        .optimize = optimize,
    });
    const minhash = b.addLibrary(.{
        .name = "minhash",
        .root_module = minhash_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(minhash);

    // AST statistics library
    const ast_stats_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/ast_stats/ast_stats.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ast_stats = b.addLibrary(.{
        .name = "ast_stats",
        .root_module = ast_stats_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(ast_stats);

    // Normalization library
    const normalization_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/preprocessing/normalization.zig"),
        .target = target,
        .optimize = optimize,
    });
    const normalization = b.addLibrary(.{
        .name = "normalization",
        .root_module = normalization_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(normalization);

    // Tokenizer library
    const tokenizer_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/tokenization/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tokenizer = b.addLibrary(.{
        .name = "tokenizer",
        .root_module = tokenizer_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(tokenizer);

    // Features library
    const features_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/features/features.zig"),
        .target = target,
        .optimize = optimize,
    });
    const features = b.addLibrary(.{
        .name = "features",
        .root_module = features_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(features);

    // Tests for Zig modules
    const test_step = b.step("test", "Run Zig unit tests");

    const shingling_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/shingling/shingling.zig"),
        .target = target,
        .optimize = optimize,
    });
    const shingling_tests = b.addTest(.{
        .name = "shingling_test",
        .root_module = shingling_test_mod,
    });
    const run_shingling_tests = b.addRunArtifact(shingling_tests);
    test_step.dependOn(&run_shingling_tests.step);

    const minhash_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/lsh/minhash.zig"),
        .target = target,
        .optimize = optimize,
    });
    const minhash_tests = b.addTest(.{
        .name = "minhash_test",
        .root_module = minhash_test_mod,
    });
    const run_minhash_tests = b.addRunArtifact(minhash_tests);
    test_step.dependOn(&run_minhash_tests.step);

    const ast_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/ast_stats/ast_stats.zig"),
        .target = target,
        .optimize = optimize,
    });
    const ast_tests = b.addTest(.{
        .name = "ast_stats_test",
        .root_module = ast_test_mod,
    });
    const run_ast_tests = b.addRunArtifact(ast_tests);
    test_step.dependOn(&run_ast_tests.step);

    const normalization_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/preprocessing/normalization.zig"),
        .target = target,
        .optimize = optimize,
    });
    const normalization_tests = b.addTest(.{
        .name = "normalization_test",
        .root_module = normalization_test_mod,
    });
    const run_normalization_tests = b.addRunArtifact(normalization_tests);
    test_step.dependOn(&run_normalization_tests.step);

    const tokenizer_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/tokenization/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tokenizer_tests = b.addTest(.{
        .name = "tokenizer_test",
        .root_module = tokenizer_test_mod,
    });
    const run_tokenizer_tests = b.addRunArtifact(tokenizer_tests);
    test_step.dependOn(&run_tokenizer_tests.step);

    const features_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/features/features.zig"),
        .target = target,
        .optimize = optimize,
    });
    const features_tests = b.addTest(.{
        .name = "features_test",
        .root_module = features_test_mod,
    });
    const run_features_tests = b.addRunArtifact(features_tests);
    test_step.dependOn(&run_features_tests.step);
}
