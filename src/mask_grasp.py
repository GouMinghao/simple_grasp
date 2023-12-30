import numpy as np
import open3d as o3d

from .geometry import Plane, restricted_uv2xyz, get_plane_frame_trans
from .pca import points_pca, uvs_from_mask


def gen_grasp_mono(mask: np.array, camK: np.array, plane: Plane) -> np.array:
    """generate grasp pose from given parameters

    Args:
        # rgb (np.array): rgb image
        # depth (np.array): depth image
        mask (np.array): object mask in rgb
        camK (np.array): camera intrinsics
        plane (Plane): the plane

    Returns:
        np.array: grasp transformation matrix
    """
    # calculate uvs from mask
    uvs = uvs_from_mask(mask)

    # xyzs in camera frame
    xyzs_c = restricted_uv2xyz(uvs, camK=camK, plane=plane)
    mask_pts = o3d.geometry.PointCloud()
    mask_pts.points = o3d.utility.Vector3dVector(xyzs_c)
    mask_pts.paint_uniform_color(np.array((0, 0, 1.0)))

    # get plane frame transformations
    plane_trans = get_plane_frame_trans(plane)

    # calculate xyzs in plane frame
    mask_pts.transform(np.linalg.inv(plane_trans))

    center_2d_p, main_direct_2d_p, sub_direct_2d_p = points_pca(
        np.asarray(mask_pts.points)[:, :2]
    )

    center_3d_p = np.hstack(
        (
            center_2d_p,
            np.array(
                0,
            ),
        )
    )
    main_direct_3d_p = np.hstack(
        (main_direct_2d_p / np.linalg.norm(main_direct_2d_p), np.array((0,)))
    )
    sub_direct_3d_p = np.hstack(
        (sub_direct_2d_p / np.linalg.norm(sub_direct_2d_p), np.array((0,)))
    )
    norm_direct_3d_p = np.cross(main_direct_3d_p, sub_direct_3d_p)
    grasp_trans_p = np.vstack(
        (
            np.hstack(
                (
                    np.vstack((main_direct_3d_p, sub_direct_3d_p, norm_direct_3d_p)).T,
                    center_3d_p.reshape((3, 1)),
                )
            ),
            np.array((0, 0, 0, 1)),
        )
    )

    # grasp normal should point the plane
    if grasp_trans_p[2, 2] > 0:
        grasp_trans_p[:3, :3] *= -1

    # vis
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.02)
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    grasp_frame.transform(grasp_trans_p)
    o3d.visualization.draw_geometries([frame, mask_pts, grasp_frame])
    return grasp_trans_p
