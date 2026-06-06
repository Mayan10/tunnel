import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from tunnel.audio import analyze_audio_file


class AudioAnalysisTest(unittest.TestCase):
    def test_analyzes_generated_wav(self):
        sample_rate = 11_025
        seconds = 4
        samples = array("h")
        for index in range(sample_rate * seconds):
            beat = 1.0 if index % (sample_rate // 2) < 900 else 0.25
            tone = math.sin(2 * math.pi * 220 * index / sample_rate)
            samples.append(int(beat * tone * 12_000))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)

        try:
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(samples.tobytes())

            features = analyze_audio_file(path)

            self.assertGreater(features.energy, 0)
            self.assertGreaterEqual(features.brightness, 0)
            self.assertLessEqual(features.brightness, 1)
            self.assertGreater(features.analyzed_seconds, 3)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
