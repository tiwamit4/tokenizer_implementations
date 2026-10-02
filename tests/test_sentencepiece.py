from pathlib import Path
import tempfile
import unittest

from SentencePiece.code.tokenizer import SentencePieceTokenizer
from SentencePiece.code.training import train_sentencepiece


class SentencePieceTests(unittest.TestCase):
    CORPUS = [
        "hello world hello tokenizer",
        "SentencePiece uses raw sentences.",
        "नमस्ते दुनिया। こんにちは世界。",
    ] * 4

    def train(self, folder, model_type):
        return SentencePieceTokenizer.train(
            iter(self.CORPUS), Path(folder) / model_type,
            model_type, 64, hard_vocab_limit=False)

    def test_unigram_and_bpe_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            for model_type in ("unigram", "bpe"):
                tokenizer = self.train(folder, model_type)
                text = "hello world नमस्ते"
                ids = tokenizer.encode(text)
                self.assertEqual(tokenizer.decode(ids), text)
                self.assertEqual(
                    tokenizer.decode(tokenizer.tokenize(text)), text)
                self.assertEqual(tokenizer.id_to_piece(ids[0]),
                                 tokenizer.tokenize(text)[0])
                self.assertEqual(tokenizer.piece_to_id("<unk>"), 0)
                self.assertEqual(tokenizer.piece_to_id("<pad>"), 3)

    def test_model_files_and_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = self.train(folder, "unigram")
            self.assertTrue(tokenizer.model_file.is_file())
            self.assertTrue(tokenizer.model_file.with_suffix(".vocab").is_file())
            restored = SentencePieceTokenizer(tokenizer.model_file)
            self.assertEqual(restored.encode("hello"), tokenizer.encode("hello"))
            with self.assertRaises(FileExistsError):
                self.train(folder, "unigram")

    def test_sampling_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            tokenizer = self.train(folder, "unigram")
            self.assertIsInstance(tokenizer.sample("hello world"), list)
            with self.assertRaises(ValueError):
                tokenizer.sample("hello", alpha=0)
            with self.assertRaises(ValueError):
                tokenizer.id_to_piece(-1)
        for args in [([], "x", "unigram", 10), (["x"], "x", "bad", 10)]:
            with tempfile.TemporaryDirectory() as folder:
                values = list(args)
                values[1] = Path(folder) / values[1]
                with self.assertRaises(ValueError):
                    train_sentencepiece(*values)


if __name__ == "__main__":
    unittest.main()
