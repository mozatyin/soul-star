"""soul_star._video — MP4 export via imageio."""

import imageio
import numpy as np
from pathlib import Path
from PIL import Image


def make_mp4(cinema_frames, output_path, fps=24):
    """
    Write a list of PIL Images or numpy arrays to an MP4 file.

    Parameters
    ----------
    cinema_frames : list[PIL.Image | np.ndarray]  — RGB frames (any size, must be consistent)
    output_path : str | Path
    fps : int — default 24 (cinematic)

    Returns
    -------
    Path — output_path as Path object
    """
    output_path = Path(output_path)
    arrays = []
    for f in cinema_frames:
        if isinstance(f, Image.Image):
            arrays.append(np.array(f.convert('RGB')))
        else:
            arrays.append(np.asarray(f))
    imageio.mimwrite(str(output_path), arrays, fps=fps, codec='libx264',
                     quality=8, macro_block_size=None)
    return output_path


def make_mp4_from_paths(frame_paths, output_path, fps=24):
    """Load PIL Images from paths (sorted) and write MP4."""
    frames = [Image.open(p).convert('RGB') for p in sorted(frame_paths)]
    return make_mp4(frames, output_path, fps=fps)
