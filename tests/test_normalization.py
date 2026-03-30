"""Tests for the normalization pipeline."""

import pytest

from src.bindings.normalization import NormalizationLib


@pytest.fixture(scope="module")
def norm():
    return NormalizationLib()


class TestRemoveComments:
    def test_single_line_comment(self, norm):
        src = "int x = 5; // this is a comment\nint y = 10;"
        result = norm.remove_comments(src)
        assert "//" not in result
        assert "int x = 5;" in result
        assert "int y = 10;" in result

    def test_multi_line_comment(self, norm):
        src = "int x = 5; /* block\ncomment */ int y = 10;"
        result = norm.remove_comments(src)
        assert "/*" not in result
        assert "*/" not in result
        assert "int x = 5;" in result
        assert "int y = 10;" in result

    def test_preserves_string_with_comment_syntax(self, norm):
        src = 'String s = "// not a comment"; // real comment'
        result = norm.remove_comments(src)
        assert '"// not a comment"' in result
        assert "// real comment" not in result

    def test_preserves_char_literal(self, norm):
        src = "char c = '/'; // comment"
        result = norm.remove_comments(src)
        assert "'/'" in result

    def test_escaped_quote_in_string(self, norm):
        src = r'String s = "hello \"world\""; // comment'
        result = norm.remove_comments(src)
        assert '// comment' not in result

    def test_empty_input(self, norm):
        result = norm.remove_comments("")
        assert result == ""


class TestNormalizeWhitespace:
    def test_collapses_spaces(self, norm):
        src = "int    x   =    5;"
        result = norm.normalize_whitespace(src)
        assert result == "int x = 5;"

    def test_removes_blank_lines(self, norm):
        src = "int x = 5;\n\n\n\nint y = 10;"
        result = norm.normalize_whitespace(src)
        assert result == "int x = 5;\nint y = 10;"

    def test_tabs_to_spaces(self, norm):
        src = "\tint\tx\t=\t5;"
        result = norm.normalize_whitespace(src)
        assert result == "int x = 5;"

    def test_carriage_returns(self, norm):
        src = "int x = 5;\r\nint y = 10;"
        result = norm.normalize_whitespace(src)
        assert "\r" not in result

    def test_empty_input(self, norm):
        result = norm.normalize_whitespace("")
        assert result == ""


class TestNormalizeSource:
    def test_full_pipeline(self, norm):
        src = "// header\nint x = 5;  /* block\ncomment */  int y = 10;\n\n\n"
        result = norm.normalize_source(src)
        assert "//" not in result
        assert "/*" not in result
        assert "int x = 5;" in result
        assert "int y = 10;" in result

    def test_real_java_method(self, norm):
        src = """// Insert helper method
private void insertHelper(ForceItem p, QuadTreeNode n, float x1, float y1, float x2, float y2) {
    float x = p.location[0], y = p.location[1]; // get coordinates
    float splitx = (x1 + x2) / 2;
    /* calculate midpoint */
    float splity = (y1 + y2) / 2;


    int i = (x >= splitx ? 1 : 0) + (y >= splity ? 2 : 0);
}
"""
        result = norm.normalize_source(src)
        assert "//" not in result
        assert "/*" not in result
        assert "float x = p.location[0]" in result
        assert "float splitx = (x1 + x2) / 2;" in result
        assert "\n\n\n" not in result

    def test_preserves_code_structure(self, norm):
        src = "if (x > 0) {\n    return x;\n} else {\n    return -x;\n}"
        result = norm.normalize_source(src)
        assert "if (x > 0)" in result
        assert "return x;" in result
        assert "return -x;" in result


class TestComputeNormImpact:
    def test_impact_metrics(self, norm):
        orig = "int x = 5;\n\n\n// comment\nint y = 10;\n"
        norm_src = norm.normalize_source(orig)
        impact = norm.compute_norm_impact(orig, norm_src)
        assert impact["original_lines"] > impact["normalized_lines"]
        assert impact["blank_lines_removed"] > 0
