import unittest

from tunnel.audio import AudioFeatures
from tunnel.embedding import EMBEDDING_DIMS, embed_tracks
from tunnel.types import Track


class EmbeddingTest(unittest.TestCase):
    def test_embed_tracks_returns_empty_without_audio_features(self):
        track = Track(id="a", name="A")
        self.assertEqual(embed_tracks([track], {}), {})

    def test_embed_tracks_skips_tracks_without_matching_audio(self):
        tracks = [Track(id="a", name="A"), Track(id="b", name="B")]
        audio = {"a": AudioFeatures(bpm=120, energy=0.5, brightness=0.5, dynamics=0.5, analyzed_seconds=30.0)}
        embeddings = embed_tracks(tracks, audio)
        self.assertNotIn("b", embeddings)

    def test_bundled_model_produces_unit_embeddings(self):
        track = Track(id="a", name="A", bpm=120, duration=200)
        audio = AudioFeatures(
            bpm=120,
            energy=0.6,
            brightness=0.5,
            dynamics=0.4,
            analyzed_seconds=30.0,
            bands=(0.4, 0.3, 0.2, 0.1),
        )

        embeddings = embed_tracks([track], {"a": audio})

        self.assertIn("a", embeddings)
        vector = embeddings["a"]
        self.assertEqual(len(vector), EMBEDDING_DIMS)
        norm = sum(value * value for value in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_bundled_model_separates_dissimilar_archetypes(self):
        ambient = Track(id="ambient", name="Ambient", bpm=70, duration=240)
        ambient_audio = AudioFeatures(
            bpm=70, energy=0.15, brightness=0.2, dynamics=0.2, analyzed_seconds=30.0, bands=(0.55, 0.25, 0.13, 0.07)
        )
        dance = Track(id="dance", name="Dance", bpm=128, duration=200)
        dance_audio = AudioFeatures(
            bpm=128, energy=0.8, brightness=0.7, dynamics=0.28, analyzed_seconds=30.0, bands=(0.3, 0.22, 0.25, 0.23)
        )

        embeddings = embed_tracks([ambient, dance], {"ambient": ambient_audio, "dance": dance_audio})

        similarity = sum(a * b for a, b in zip(embeddings["ambient"], embeddings["dance"], strict=True))
        self.assertLess(similarity, 0.5)

    def test_batched_embedding_matches_individual_calls_per_track(self):
        specs = [
            (70, 0.15, 0.2, 0.2, (0.55, 0.25, 0.13, 0.07)),
            (128, 0.8, 0.7, 0.28, (0.3, 0.22, 0.25, 0.23)),
            (95, 0.4, 0.45, 0.4, (0.4, 0.3, 0.2, 0.1)),
            (150, 0.9, 0.75, 0.3, (0.28, 0.22, 0.27, 0.23)),
        ]
        tracks = []
        audio_features = {}
        for index, (bpm, energy, brightness, dynamics, bands) in enumerate(specs):
            track_id = f"track-{index}"
            tracks.append(Track(id=track_id, name=track_id, bpm=bpm, duration=200))
            audio_features[track_id] = AudioFeatures(
                bpm=bpm, energy=energy, brightness=brightness, dynamics=dynamics, analyzed_seconds=30.0, bands=bands
            )

        batched = embed_tracks(tracks, audio_features)
        self.assertEqual(len(batched), len(tracks))

        for track in tracks:
            individual = embed_tracks([track], {track.id: audio_features[track.id]})
            for batched_value, individual_value in zip(batched[track.id], individual[track.id], strict=True):
                self.assertAlmostEqual(batched_value, individual_value, places=4)


if __name__ == "__main__":
    unittest.main()
