import csv
from pathlib import Path
import random
import tempfile
import unittest

from BPE.code.data import iter_texts
from BPE.code.model_io import load_model, save_model
from BPE.code.optimized_training import train_bpe_optimized
from BPE.code.tokenizer import decode, encode_text
from BPE.code.training import train_bpe


class ScalingTests(unittest.TestCase):
    def test_matches_simple_training(self):
        randomizer = random.Random(42)
        for _ in range(30):
            corpus = ["".join(randomizer.choices("abc", k=randomizer.randint(1, 12)))
                      for _ in range(30)]
            self.assertEqual(train_bpe(corpus, 30), train_bpe_optimized(iter(corpus), 30))

    def test_csv_streaming_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows([["text"], ["low lower"], [""], ["lowest"], ["other"]])
            texts = list(iter_texts([path], max_texts=2))
            self.assertEqual(texts, ["low lower", "lowest"])
            merges, tokens = train_bpe_optimized(texts, 20)
            self.assertEqual(decode(encode_text("low lower", merges, tokens)), "low lower")
            model = Path(folder) / "model.json"
            save_model(model, merges, tokens, 20)
            self.assertEqual(load_model(model), (merges, tokens, 20))

    def test_invalid_input(self):
        for corpus, count in [([], 10), (["abc"], -1)]:
            with self.assertRaises(ValueError):
                train_bpe_optimized(corpus, count)


if __name__ == "__main__":
    unittest.main()
