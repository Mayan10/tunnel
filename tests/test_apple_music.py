import unittest

from tunnel.apple_music import AppleMusicError, _ordered_source_indexes
from tunnel.types import Track


class AppleMusicWriteTest(unittest.TestCase):
    def test_ordered_source_indexes_preserve_duplicates(self):
        tracks = [
            Track(id="same", name="Repeat", source_index=3),
            Track(id="same", name="Repeat", source_index=1),
            Track(id="other", name="Other", source_index=2),
        ]

        self.assertEqual(_ordered_source_indexes(tracks), ["3", "1", "2"])

    def test_missing_source_index_is_rejected(self):
        with self.assertRaises(AppleMusicError):
            _ordered_source_indexes([Track(id="x", name="Missing")])


if __name__ == "__main__":
    unittest.main()
