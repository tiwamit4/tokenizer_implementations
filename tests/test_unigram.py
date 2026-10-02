from pathlib import Path
import tempfile
import unittest

from Unigram.code.segmentation import viterbi
from Unigram.code.tokenizer import UnigramTokenizer
from Unigram.code.training import train_unigram


class UnigramTests(unittest.TestCase):
    CORPUS = ["hug " * 10, "pug " * 5, "pun " * 12, "bun " * 4, "hugs " * 5]

    def test_viterbi_selects_lowest_cost_path(self):
        pieces, cost = viterbi("unhug", {"u": 2, "n": 2, "un": 1, "h": 2, "hug": 1})
        self.assertEqual(pieces, ["un", "hug"])
        self.assertEqual(cost, 2)

    def test_training_is_deterministic_and_keeps_characters(self):
        first = train_unigram(iter(self.CORPUS), 30, 60)
        self.assertEqual(first, train_unigram(self.CORPUS, 30, 60))
        tokenizer = UnigramTokenizer(first)
        for text in self.CORPUS:
            self.assertNotIn("<unk>", tokenizer.tokenize(text))
        self.assertEqual(len(tokenizer.vocabulary), 30)

    def test_unknown_save_and_load(self):
        tokenizer = UnigramTokenizer.train(self.CORPUS, 30, 60)
        self.assertEqual(tokenizer.tokenize("zebra"), ["<unk>"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("hug zebra")), "hug <unk>")
        self.assertEqual(tokenizer.decode(tokenizer.encode("hug pug")), "hug pug")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            tokenizer.save(path)
            restored = UnigramTokenizer.load(path)
            self.assertEqual(restored.encode("hug pug"), tokenizer.encode("hug pug"))
            with self.assertRaises(FileExistsError):
                tokenizer.save(path)
        with self.assertRaises(ValueError):
            tokenizer.decode([-1])

    def test_invalid_training_settings(self):
        for args in [([], 30, 60), (["abc"], 2, 20), (["abc"], 10, 2)]:
            with self.assertRaises(ValueError):
                train_unigram(*args)


if __name__ == "__main__":
    unittest.main()
