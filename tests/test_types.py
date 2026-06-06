import unittest

from tunnel.types import Track


class TrackTypesTest(unittest.TestCase):
    def test_zero_music_values_are_treated_as_unknown(self):
        track = Track.from_dict(
            {
                "name": "No Metadata",
                "bpm": 0,
                "year": 0,
                "duration": 0,
                "played_count": 0,
            }
        )

        self.assertIsNone(track.bpm)
        self.assertIsNone(track.year)
        self.assertIsNone(track.duration)
        self.assertEqual(track.played_count, 0)


if __name__ == "__main__":
    unittest.main()
