from pathlib import Path
import tempfile
import unittest

from ByteLevelBPE.code.bytes_mapping import BYTE_DECODER, BYTE_ENCODER
from ByteLevelBPE.code.tokenizer import ByteLevelBPETokenizer
from ByteLevelBPE.code.training import train_byte_level_bpe


class ByteLevelBPETests(unittest.TestCase):
    def test_byte_mapping_is_bijective(self):
        self.assertEqual(len(BYTE_ENCODER), 256)
        self.assertEqual({BYTE_DECODER[value] for value in BYTE_DECODER},
                         set(range(256)))

    def test_every_utf8_text_round_trips_without_unknown(self):
        corpus = ["hello hello world", "नमस्ते 👋 café", "tabs\tand\nlines"]
        tokenizer = ByteLevelBPETokenizer.train(corpus, 300)
        for text in corpus + ["unseen 🧪 中文\n new text"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertNotIn("<unk>", tokenizer.vocabulary)

    def test_training_is_deterministic(self):
        corpus = ["low lower lowest", "low low", "नमस्ते"]
        first = train_byte_level_bpe(iter(corpus), 280)
        self.assertEqual(first, train_byte_level_bpe(corpus, 280))
        # A small corpus can become fully merged before reaching the target.
        self.assertGreater(len(first), 0)
        self.assertLessEqual(len(first), 23)
        with self.assertRaises(ValueError):
            train_byte_level_bpe([], 280)
        with self.assertRaises(ValueError):
            train_byte_level_bpe(["text"], 256)

    def test_save_load_and_validation(self):
        tokenizer = ByteLevelBPETokenizer.train(["hello hello"], 270)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            tokenizer.save(path)
            restored = ByteLevelBPETokenizer.load(path)
            text = "hello 👋"
            self.assertEqual(restored.encode(text), tokenizer.encode(text))
            self.assertEqual(restored.decode(restored.encode(text)), text)
            with self.assertRaises(FileExistsError):
                tokenizer.save(path)
        with self.assertRaises(ValueError):
            tokenizer.decode([-1])


if __name__ == "__main__":
    unittest.main()
