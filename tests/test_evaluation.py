import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gb_tetris_rl.agent.evaluation import _write_gif


class GifRecordingTests(unittest.TestCase):
    def test_writes_all_supplied_frames(self) -> None:
        frames = [np.full((12, 10, 3), fill_value=shade, dtype=np.uint8) for shade in (0, 127, 255)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            recording_path = Path(temporary_directory) / "agent.gif"

            _write_gif(frames, recording_path, frame_duration_ms=50)

            with Image.open(recording_path) as recorded_gif:
                self.assertEqual(recorded_gif.n_frames, 3)
                self.assertEqual(recorded_gif.info["duration"], 50)


if __name__ == "__main__":
    unittest.main()
