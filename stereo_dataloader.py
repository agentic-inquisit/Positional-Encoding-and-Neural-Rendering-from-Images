import os
import os.path as osp

import imageio.v2 as imageio
import numpy as np


def load_middlebury_data(path):
    """Load a Middlebury multi-view-stereo sequence (e.g. templeRing).

    `path` must contain the per-view images plus a `*_par.txt` calibration
    file where each non-header line is:
        imgname.png  K (9 values, row-major)  R (9 values, row-major)  T (3 values)
    following the convention p_cam = R @ p_world + T.

    Returns a list of dicts, one per view, with keys "rgb", "R", "T", "K".
    """
    par_files = [f for f in os.listdir(path) if f.endswith("_par.txt")]
    if not par_files:
        raise FileNotFoundError(f"No *_par.txt calibration file found in {path}")
    par_path = osp.join(path, par_files[0])

    with open(par_path) as f:
        lines = f.read().strip().split("\n")

    num_views = int(lines[0])
    data = []
    for line in lines[1 : 1 + num_views]:
        tokens = line.split()
        img_name = tokens[0]
        values = list(map(float, tokens[1:]))
        K = np.array(values[0:9]).reshape(3, 3)
        R = np.array(values[9:18]).reshape(3, 3)
        T = np.array(values[18:21])

        rgb = imageio.imread(osp.join(path, img_name))
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        data.append({"rgb": rgb, "R": R, "T": T, "K": K})

    return data
