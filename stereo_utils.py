import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def viz_camera_poses(data, scale=0.05):
    """Plot camera centers and viewing directions in world coordinates.

    `data` is the list of view dicts returned by load_middlebury_data,
    each with "R" (world->camera rotation) and "T" (world->camera
    translation), following p_cam = R @ p_world + T.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    centers = []
    for view in data:
        R, T = view["R"], view["T"]
        center = -R.T @ T
        centers.append(center)
        forward = R.T @ np.array([0, 0, 1])
        ax.quiver(*center, *forward, length=scale, color="r")

    centers = np.array(centers)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="b", s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera poses")
    plt.tight_layout()
    return fig
