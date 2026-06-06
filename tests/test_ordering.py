import unittest

from tunnel.model import LocalFlowModel
from tunnel.ordering import OrderingConfig, order_tracks
from tunnel.types import Track


class OrderingTest(unittest.TestCase):
    def test_ordering_keeps_every_track_once(self):
        tracks = [
            Track(id="a", name="A", genre="Ambient", bpm=72),
            Track(id="b", name="B", genre="Dance", bpm=126),
            Track(id="c", name="C", genre="Indie Pop", bpm=96),
            Track(id="d", name="D", genre="Techno", bpm=132),
        ]

        ordered = order_tracks(tracks, config=OrderingConfig(swap_passes=1))

        self.assertEqual(sorted(track.id for track in ordered.tracks), ["a", "b", "c", "d"])
        self.assertEqual(len({track.id for track in ordered.tracks}), len(tracks))

    def test_energy_curve_starts_with_lower_energy_track(self):
        tracks = [
            Track(id="high", name="Peak", genre="Techno", bpm=134),
            Track(id="low", name="Open", genre="Ambient", bpm=72),
            Track(id="mid", name="Cruise", genre="Pop", bpm=104),
        ]

        ordered = order_tracks(tracks, config=OrderingConfig(swap_passes=0))

        self.assertEqual(ordered.tracks[0].id, "low")

    def test_transition_cost_is_lower_for_similar_tracks(self):
        model = LocalFlowModel()
        left = Track(id="left", name="Left", artist="A", album="One", genre="House", bpm=124, year=2022)
        similar = Track(id="similar", name="Similar", artist="A", album="One", genre="House", bpm=126, year=2022)
        different = Track(id="different", name="Different", artist="Z", album="Far", genre="Folk", bpm=78, year=1980)

        left_features = model.features_for(left)
        similar_cost = model.transition(left, similar, left_features, model.features_for(similar)).total
        different_cost = model.transition(left, different, left_features, model.features_for(different)).total

        self.assertLess(similar_cost, different_cost)


if __name__ == "__main__":
    unittest.main()
