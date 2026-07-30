"""Unit tests for the benchmark/supervised data plumbing (common.py, prepare_data.py).

Run: python -m pytest tests/ -q   (or: python -m unittest)
Stdlib-only — no torch/spacy/transformers needed; the offset math and split
logic are exercised without touching any model.
"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "benchmark", "supervised"))

import common  # noqa: E402
import prepare_data  # noqa: E402


def ent(label, text, start, end):
    return {"label": label, "text": text, "start": start, "end": end}


class TestTokenize(unittest.TestCase):
    def test_offsets_index_into_original_text(self):
        text = "My 3yo-son loves it!"
        for tok, s, e in common.tokenize(text):
            self.assertEqual(text[s:e], tok)

    def test_words_and_punctuation(self):
        toks = [t for t, _, _ in common.tokenize("It's great... <br />")]
        self.assertEqual(toks, ["It", "'", "s", "great", ".", ".", ".",
                                "<", "br", "/", ">"])


class TestBioConversion(unittest.TestCase):
    def test_exact_round_trip(self):
        text = "My great grandson loves this game."
        entities = [ent("FAM_KIN", "great grandson", 3, 17)]
        tokens = common.tokenize(text)
        tags, adjusted, dropped = common.spans_to_bio(tokens, entities)
        self.assertEqual((adjusted, dropped), (0, 0))
        self.assertEqual(tags[1:3], ["B-FAM_KIN", "I-FAM_KIN"])
        self.assertEqual(common.bio_to_spans(text, tokens, tags), entities)

    def test_misaligned_span_is_expanded_to_token_boundaries(self):
        text = "my grandson here"
        tokens = common.tokenize(text)
        tags, adjusted, dropped = common.spans_to_bio(
            tokens, [ent("FAM_KIN", "grands", 3, 9)]  # ends mid-word
        )
        self.assertEqual((adjusted, dropped), (1, 0))
        spans = common.bio_to_spans(text, tokens, tags)
        self.assertEqual(spans, [ent("FAM_KIN", "grandson", 3, 11)])

    def test_span_matching_no_token_is_dropped(self):
        text = "a  b"
        tokens = common.tokenize(text)
        tags, _, dropped = common.spans_to_bio(
            tokens, [ent("FAM_KIN", " ", 1, 2)]  # whitespace only
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(tags, ["O", "O"])

    def test_overlapping_second_span_is_dropped(self):
        text = "my great grandson"
        tokens = common.tokenize(text)
        tags, _, dropped = common.spans_to_bio(tokens, [
            ent("FAM_KIN", "great grandson", 3, 17),
            ent("GEN_NOUN", "grandson", 9, 17),
        ])
        self.assertEqual(dropped, 1)
        self.assertEqual(tags, ["O", "B-FAM_KIN", "I-FAM_KIN"])

    def test_adjacent_same_label_entities_stay_separate(self):
        text = "son daughter"
        tokens = common.tokenize(text)
        tags, _, _ = common.spans_to_bio(tokens, [
            ent("FAM_KIN", "son", 0, 3),
            ent("FAM_KIN", "daughter", 4, 12),
        ])
        self.assertEqual(tags, ["B-FAM_KIN", "B-FAM_KIN"])
        self.assertEqual(len(common.bio_to_spans(text, tokens, tags)), 2)

    def test_decode_tolerates_orphan_inside_tag(self):
        text = "my son here"
        tokens = common.tokenize(text)
        spans = common.bio_to_spans(text, tokens, ["O", "I-FAM_KIN", "O"])
        self.assertEqual(spans, [ent("FAM_KIN", "son", 3, 6)])


class TestSpanMicroF1(unittest.TestCase):
    def test_perfect_and_partial(self):
        gold = [[ent("FAM_KIN", "son", 3, 6)], []]
        self.assertEqual(common.span_micro_f1(gold, gold), (1.0, 1.0, 1.0))
        pred = [[ent("FAM_KIN", "son", 3, 6), ent("GEN_NOUN", "x", 0, 1)], []]
        p, r, f1 = common.span_micro_f1(gold, pred)
        self.assertEqual((p, r), (0.5, 1.0))
        self.assertAlmostEqual(f1, 2 / 3)


def make_examples(n_products=40, rows_per_product=5):
    """Synthetic gold rows: product i, some with entities."""
    examples = []
    for i in range(n_products):
        for j in range(rows_per_product):
            row_id = f"B{i:09d}_{j:05d}"
            if (i + j) % 3 == 0:
                text = "my son loves this"
                entities = [ent("KINSHIP", "son", 3, 6)]
            elif (i + j) % 7 == 0:
                text = "my 3rd grader uses it"
                entities = [ent("MINOR", "3rd grader", 3, 13)]
            else:
                text = "works fine"
                entities = []
            examples.append(common.Example(
                row_id=row_id, text=text, entities=entities,
                raw_row={
                    "row_id": row_id, "raw_text": text, "tagged_text": text,
                    "num_entities": str(len(entities)),
                    "entities_json": json.dumps(entities),
                },
            ))
    return examples


class TestSplit(unittest.TestCase):
    def test_partition_is_complete_and_disjoint(self):
        examples = make_examples()
        splits = common.split_examples(examples)
        ids = [ex.row_id for split in splits.values() for ex in split]
        self.assertEqual(sorted(ids), sorted(ex.row_id for ex in examples))
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_product_group_straddles_splits(self):
        splits = common.split_examples(make_examples())
        owner = {}
        for name, split in splits.items():
            for ex in split:
                key = common.group_key(ex.row_id)
                self.assertEqual(owner.setdefault(key, name), name,
                                 f"group {key} appears in two splits")

    def test_deterministic_for_fixed_seed(self):
        a = common.split_examples(make_examples(), seed=13)
        b = common.split_examples(make_examples(), seed=13)
        for name in a:
            self.assertEqual([ex.row_id for ex in a[name]],
                             [ex.row_id for ex in b[name]])

    def test_fractions_roughly_hold(self):
        splits = common.split_examples(make_examples(n_products=100))
        total = sum(len(s) for s in splits.values())
        self.assertGreater(len(splits["train"]) / total, 0.7)
        self.assertGreater(len(splits["dev"]), 0)
        self.assertGreater(len(splits["test"]), 0)

    def test_rare_label_present_in_every_split(self):
        splits = common.split_examples(make_examples(n_products=100))
        for name, split in splits.items():
            labels = {e["label"] for ex in split for e in ex.entities}
            self.assertIn("MINOR", labels, f"MINOR missing from {name}")


class TestGlinerFormat(unittest.TestCase):
    def test_record_uses_inclusive_token_indices_and_phrases(self):
        ex = common.Example(
            row_id="X_1", text="My great grandson loves this game.",
            entities=[ent("KINSHIP", "great grandson", 3, 17)], raw_row={},
        )
        record, truncated = prepare_data.to_gliner_record(ex)
        self.assertEqual(truncated, 0)
        self.assertEqual(record["tokenized_text"][:3], ["My", "great", "grandson"])
        self.assertEqual(record["ner"], [
            [1, 2, common.GLINER_LABEL_PHRASES["KINSHIP"]],
        ])

    def test_phrase_mapping_round_trips(self):
        for code, phrase in common.GLINER_LABEL_PHRASES.items():
            self.assertEqual(common.PHRASE_TO_LABEL[phrase], code)
        self.assertEqual(set(common.GLINER_LABEL_PHRASES), set(common.LABELS))

    def test_span_beyond_truncation_is_dropped(self):
        n = prepare_data.GLINER_MAX_TOKENS
        text = "w " * n + "son"
        ex = common.Example(
            row_id="X_2", text=text,
            entities=[ent("FAM_KIN", "son", 2 * n, 2 * n + 3)], raw_row={},
        )
        record, truncated = prepare_data.to_gliner_record(ex)
        self.assertEqual(truncated, 1)
        self.assertEqual(record["ner"], [])
        self.assertEqual(len(record["tokenized_text"]), n)


class TestPrepareDataEndToEnd(unittest.TestCase):
    def test_full_pipeline_on_tiny_csv(self):
        examples = make_examples(n_products=30, rows_per_product=4)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gold = tmp / "gold.csv"
            with open(gold, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(examples[0].raw_row))
                writer.writeheader()
                for ex in examples:
                    writer.writerow(ex.raw_row)

            splits = prepare_data.prepare(gold, tmp / "data", neg_ratio=1.0)

            for name in ("train", "dev", "test"):
                self.assertTrue((tmp / "data" / "splits" / f"{name}.csv").exists())
                self.assertTrue((tmp / "data" / "bio" / f"{name}.jsonl").exists())

            # neg-ratio applies to train only
            n_pos = sum(1 for ex in splits["train"] if ex.entities)
            n_neg = sum(1 for ex in splits["train"] if not ex.entities)
            self.assertLessEqual(n_neg, n_pos)

            # BIO files must round-trip to the gold entities exactly
            with open(tmp / "data" / "bio" / "test.jsonl", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            gold_by_id = {ex.row_id: ex for ex in splits["test"]}
            self.assertEqual({r["row_id"] for r in rows}, set(gold_by_id))
            for r in rows:
                tokens = list(zip(r["tokens"], r["starts"], r["ends"]))
                spans = common.bio_to_spans(r["text"], tokens, r["tags"])
                self.assertEqual(spans, gold_by_id[r["row_id"]].entities)

            # GLiNER files parse and only contain known phrases
            with open(tmp / "data" / "gliner" / "train.json", encoding="utf-8") as f:
                records = json.load(f)
            self.assertEqual(len(records), len(splits["train"]))
            for rec in records:
                for start_tok, end_tok, phrase in rec["ner"]:
                    self.assertIn(phrase, common.PHRASE_TO_LABEL)
                    self.assertLessEqual(start_tok, end_tok)
                    self.assertLess(end_tok, len(rec["tokenized_text"]))

            # split CSVs reload cleanly through the same loader
            reloaded = common.load_examples(tmp / "data" / "splits" / "dev.csv")
            self.assertEqual([ex.row_id for ex in reloaded],
                             [ex.row_id for ex in splits["dev"]])


if __name__ == "__main__":
    unittest.main()
