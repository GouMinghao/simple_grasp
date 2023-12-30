import cv2
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from graspnetAPI import Grasp
from src.graspnet_utils.graspnet_utils import GraspNetUtils, ANN_IDS, SCENE_ID
from src.pca import mask_pca, uvs_from_mask, points_pca
from src.geometry import Plane, segment_plane, get_plane_frame_trans, restricted_uv2xyz

def gen_grasp(
        # rgb: np.array,
        depth: np.array,
        mask: np.array,
        camK: np.array,
        plane: Plane) -> Grasp:
    """generate grasp pose from given parameters

    Args:
        # rgb (np.array): rgb image
        depth (np.array): depth image
        mask (np.array): object mask in rgb
        camK (np.array): camera intrinsics
        plane (Plane): the plane 

    Returns:
        Grasp: generated grasp pose
    """
    pass

if __name__ == "__main__":
    graspnet_utils = GraspNetUtils()
    obj_ids = graspnet_utils.load_object_id_list(SCENE_ID)
    select_obj_id = obj_ids[2]
    obj_mask = graspnet_utils.load_obj_mask(select_obj_id, SCENE_ID, ANN_IDS[1])
    all_data = graspnet_utils.load_all_data(SCENE_ID, ANN_IDS[1])
    pcd = all_data["pcd"]
    rgb = all_data["rgb"]
    camK = graspnet_utils.load_camK()
    uvs = uvs_from_mask(obj_mask)


    plane, inliers = segment_plane(pcd)


    xyzs = restricted_uv2xyz(uvs, camK=camK, plane=plane)
    plane_trans = get_plane_frame_trans(plane)
    print(plane_trans)
    colors = np.asarray(pcd.colors)
    colors[inliers] = np.array((1,0,0))

    mask_pts = o3d.geometry.PointCloud()
    mask_pts.points = o3d.utility.Vector3dVector(xyzs)
    mask_pts.paint_uniform_color(np.array((0,0,1.0)))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1).transform(plane_trans)
    o3d.visualization.draw_geometries([pcd, frame, new_frame, mask_pts])


    # center, e1xy, e2xy = mask_pca(obj_mask)
    # p1 = (center + 0.01*e1xy).astype(np.int32)  # P1:[149 403]
    # p2 = (center + 0.01*e2xy).astype(np.int32)  # P2:[320 332]

    # rgb_copy = rgb.copy()
    # cv2.circle(rgb_copy, center, 6, 255, -1)  # 在PCA中心位置画一个圆圈  RGB
    # cv2.arrowedLine(rgb_copy, center, p1, (255, 0, 0), thickness=3, tipLength=0.1)  # 从 center 指向 pt1
    # cv2.arrowedLine(rgb_copy, center, p2, (255, 0, 0), thickness=3, tipLength=0.2)  # 从 center 指向 pt2
    # print("center:{}, P1:{}, P2:{}".format(center, p1, p2))
    # plt.imshow(rgb_copy)
    # plt.show()
