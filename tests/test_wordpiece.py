from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from WordPiece.code import WordPieceTokenizer, train_wordpiece
from WordPiece.code.training import get_pair_scores
from WordPiece.code.main import main
from WordPiece.code import config


class WordPieceTests(unittest.TestCase):
    def test_longest_match_and_unknown(self):
        tokenizer = WordPieceTokenizer(["[UNK]", "play", "##ing", "p", "##lay"])
        self.assertEqual(tokenizer.tokenize("playing"), ["play", "##ing"])
        self.assertEqual(tokenizer.tokenize("playz"), ["[UNK]"])
        self.assertEqual(tokenizer.tokenize("z"), ["[UNK]"])
        self.assertEqual(tokenizer.tokenize(""), [])
        self.assertEqual(tokenizer.tokenize("x" * 101), ["[UNK]"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("playing play")), "playing play")
        with self.assertRaises(ValueError):
            tokenizer.decode([-1])

    def test_pair_scores_count_repeated_symbols(self):
        scores = get_pair_scores({("a", "##a", "##a"): 2})
        self.assertEqual(scores[("a", "##a")], Fraction(1, 4))
        self.assertEqual(scores[("##a", "##a")], Fraction(1, 8))

    def test_training_is_deterministic(self):
        corpus = ["play playing played", "walk walking", "play"]
        vocabulary = train_wordpiece(iter(corpus), 50)
        self.assertEqual(vocabulary, train_wordpiece(corpus, 50))
        tokenizer = WordPieceTokenizer(vocabulary)
        for sentence in corpus:
            self.assertNotIn("[UNK]", tokenizer.tokenize(sentence))
        self.assertIn("##a", vocabulary)
        for corpus, size in [([], 50), (["abc"], 1), (["abc"], -1)]:
            with self.assertRaises(ValueError):
                train_wordpiece(corpus, size)

    def test_save_load_and_custom_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            original = WordPieceTokenizer(["[UNK]", "play", "@@ing"],
                                          lowercase=True, prefix="@@", max_chars=20)
            original.save(path)
            restored = WordPieceTokenizer.load(path)
            self.assertEqual(restored.tokenize("PLAYING"), ["play", "@@ing"])
            self.assertEqual(restored.encode("PLAYING"), original.encode("PLAYING"))
            self.assertEqual(restored.max_chars, 20)
            with self.assertRaises(FileExistsError):
                original.save(path)

    def test_config_driven_main(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            with patch.object(config, "MODEL_PATH", path), patch.object(config, "USE_DATASET", False):
                main()
                self.assertTrue(path.is_file())
                with self.assertRaises(FileExistsError):
                    main()


if __name__ == "__main__":
    unittest.main()
