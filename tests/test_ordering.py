import unittest

from tunnel.model import DEFAULT_TRANSITION_WEIGHTS, LocalFlowModel, train_playlist_model
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

    def test_metadata_only_energy_varies_within_a_single_genre(self):
        # Apple Music tags most of an artist's catalog with one genre string and
        # rarely has a BPM for streamed tracks, so a whole playlist can otherwise
        # collapse to one identical energy value with nothing left to order by.
        model = LocalFlowModel()
        short_track = Track(id="a", name="Intro", artist="Drake", genre="Hip-Hop/Rap", duration=120)
        long_track = Track(id="b", name="Outro", artist="Drake", genre="Hip-Hop/Rap", duration=320)

        energies = {model.features_for(short_track).energy, model.features_for(long_track).energy}

        self.assertEqual(len(energies), 2)

    def test_metadata_only_brightness_reflects_title_mood(self):
        model = LocalFlowModel()
        somber = Track(id="a", name="I Get Lonely", artist="Drake", genre="Hip-Hop/Rap")
        neutral = Track(id="b", name="Star67", artist="Drake", genre="Hip-Hop/Rap")

        self.assertLess(model.features_for(somber).brightness, model.features_for(neutral).brightness)

    def test_playlist_model_trains_weights_locally(self):
        tracks = [
            Track(id="a", name="A", artist="One", genre="Ambient", bpm=72, year=2018),
            Track(id="b", name="B", artist="One", genre="Downtempo", bpm=82, year=2018),
            Track(id="c", name="C", artist="Two", genre="Pop", bpm=104, year=2020),
            Track(id="d", name="D", artist="Three", genre="Dance", bpm=126, year=2022),
            Track(id="e", name="E", artist="Three", genre="House", bpm=128, year=2022),
        ]

        model = train_playlist_model(tracks)

        self.assertEqual(model.name, "audio-embedding-v1")
        self.assertNotEqual(model.weights, DEFAULT_TRANSITION_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
