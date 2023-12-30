import numpy as np
import open3d as o3d

class Plane():
    def __init__(self, a, b, c, d):
        """plane model: ax + by + cz + d = 0

        Args:
            a (float): plane parameter.
            b (float): plane parameter.
            c (float): plane parameter.
            d (float): plane parameter.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self):
        return "Plane, {:.2f}*x+{:.2f}*y+{:.2f}*z+{:.2f}=0".format(
            self.a, self.b, self.c, self.d
        )

def segment_plane(
        pcd,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
        probability=1.0) -> (Plane, np.array):
    """segment a plane from point cloud

    Args:
        pcd (o3d.geometry.PointCloud): point cloud
        distance_threshold(float): o3d dist thresh
        ransac_n(int): o3d ransac n
        num_iterations: o3d ransac iters
        probability: o3d probability

    Returns:
        Plane: segmented plane
        np.array: inliers mask
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        probability=probability
    )
    plane = Plane(
        plane_model[0],
        plane_model[1],
        plane_model[2],
        plane_model[3]
    )
    return plane, inliers

def get_plane_frame_trans(plane: Plane):
    assert not plane.c == 0
    plane_z_c = np.array([plane.a, plane.b, plane.c])
    plane_z_c = plane_z_c / np.linalg.norm(plane_z_c)
    if plane.c > 0:
        plane_z_c = -plane_z_c # normal and camera c are in opposite direction
    plane_x_c = np.cross(np.array((0, 0, 1)), plane_z_c)
    plane_x_c = plane_x_c / np.linalg.norm(plane_x_c)
    plane_y_c = np.cross(plane_z_c, plane_x_c)
    plane_t = np.array((0, 0, -plane.d / plane.c))
    trans = np.vstack((
        np.hstack((
            np.vstack((plane_x_c, plane_y_c, plane_z_c)).T,
            plane_t.reshape((3, 1))
        )),
        np.array((0, 0, 0, 1))
    ))
    return trans

def restricted_uv2xyz(uv: np.array, camK: np.array, plane: Plane) -> np.array:
    """Calculate 3d coordinates given uv, camK and plane restriction.

    Args:
        uv (np.array): [n, 2] of uvs.
        camK (np.array): [3, 3] camera intrinsic.
        plane (Plane): plane restriction.

    Returns:
        np.array: [n, 3] of xyz
    """
    fx, fy = camK[0, 0], camK[1, 1]
    cx, cy = camK[0, 2], camK[1, 2]
    z = -(plane.d / (
        plane.a * (uv[:, 0] - cx) / fx +
        plane.b * (uv[:, 1] - cy) / fy +
        plane.c))
    x = (uv[:, 0] - cx) / fx * z
    y = (uv[:, 1] - cy) / fy * z
    return np.vstack((x, y, z)).T
