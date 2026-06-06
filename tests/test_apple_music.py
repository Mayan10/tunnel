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

    def test_duplicate_source_index_is_rejected(self):
        tracks = [
            Track(id="a", name="A", source_index=1),
            Track(id="b", name="B", source_index=1),
            Track(id="c", name="C", source_index=3),
        ]

        with self.assertRaises(AppleMusicError):
            _ordered_source_indexes(tracks)

    def test_partial_source_indexes_are_rejected(self):
        tracks = [
            Track(id="a", name="A", source_index=1),
            Track(id="c", name="C", source_index=3),
        ]

        with self.assertRaises(AppleMusicError):
            _ordered_source_indexes(tracks)


if __name__ == "__main__":
    unittest.main()
