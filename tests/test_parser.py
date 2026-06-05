"""Unit tests for the deterministic regex index parser.

Run: python -m pytest tests/ -q   (or: python -m unittest)
These tests exercise the offset math with NO API calls.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.parser import (  # noqa: E402
    TagParseError,
    parse_and_verify,
    parse_tagged_text,
    strip_tags,
)


class TestParser(unittest.TestCase):
    def test_single_span_offsets(self):
        raw = "my 3yo loves this"
        tagged = "my <MINOR_AGE>3yo</MINOR_AGE> loves this"
        clean, spans = parse_tagged_text(tagged)
        self.assertEqual(clean, raw)
        self.assertEqual(len(spans), 1)
        s = spans[0]
        self.assertEqual(s.label, "MINOR_AGE")
        self.assertEqual(s.text, "3yo")
        self.assertEqual((s.start, s.end), (3, 6))
        # half-open invariant
        self.assertEqual(raw[s.start:s.end], "3yo")

    def test_multiple_spans_sequential_offsets(self):
        raw = "my 3yo son and my wife"
        tagged = (
            "my <MINOR_AGE>3yo</MINOR_AGE> <FAM_KIN>son</FAM_KIN> and my "
            "<GEN_NOUN>wife</GEN_NOUN>"
        )
        clean, spans = parse_tagged_text(tagged)
        self.assertEqual(clean, raw)
        self.assertEqual([s.label for s in spans], ["MINOR_AGE", "FAM_KIN", "GEN_NOUN"])
        for s in spans:
            self.assertEqual(raw[s.start:s.end], s.text)

    def test_demographic_compound_single_span(self):
        raw = "my 16-year-old girl"
        tagged = "my <MINOR_AGE>16-year-old girl</MINOR_AGE>"
        clean, spans = parse_tagged_text(tagged)
        self.assertEqual(clean, raw)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "16-year-old girl")

    def test_no_tags(self):
        raw = "solid build quality, five stars."
        clean, spans = parse_tagged_text(raw)
        self.assertEqual(clean, raw)
        self.assertEqual(spans, [])

    def test_strip_tags_matches_clean(self):
        tagged = "a <FAM_KIN>nephew</FAM_KIN> and a <GEN_NOUN>hubby</GEN_NOUN>"
        clean, _ = parse_tagged_text(tagged)
        self.assertEqual(strip_tags(tagged), clean)

    def test_mismatched_tags_raise(self):
        with self.assertRaises(TagParseError):
            parse_tagged_text("my <MINOR_AGE>3yo</FAM_KIN>")

    def test_unclosed_tag_raises(self):
        with self.assertRaises(TagParseError):
            parse_tagged_text("my <MINOR_AGE>3yo loves this")

    def test_stray_close_raises(self):
        with self.assertRaises(TagParseError):
            parse_tagged_text("my 3yo</MINOR_AGE>")

    def test_parse_and_verify_detects_mutation(self):
        raw = "my 12yo"
        # annotator illegally expanded 12yo -> 12-year-old
        tagged = "my <MINOR_AGE>12-year-old</MINOR_AGE>"
        with self.assertRaises(TagParseError):
            parse_and_verify(tagged, raw)

    def test_parse_and_verify_ok(self):
        raw = "my newborn twins"
        tagged = "my <MINOR_AGE>newborn twins</MINOR_AGE>"
        spans = parse_and_verify(tagged, raw)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "newborn twins")


if __name__ == "__main__":
    unittest.main()
