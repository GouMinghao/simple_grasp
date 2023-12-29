import cv2
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from graspnetAPI import Grasp
from src.graspnet_utils.graspnet_utils import GraspNetUtils, ANN_IDS, SCENE_ID

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

def gen_grasp(
        rgb: np.array,
        depth: np.array,
        mask: np.array,
        camK: np.array,
        plane: Plane) -> Grasp:
    """generate grasp pose from given parameters

    Args:
        rgb (np.array): rgb image
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

    contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # OpenCV4~
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]
    ptsXY = np.squeeze(cnt).astype(np.float64)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(ptsXY, np.array([]))
    print("mean:{}, eigenvalues:{}".format(mean.round(1), eigenvalues[:,0].round(2)))
    center = mean[0, :].astype(int)  # 近似作为目标的中心 [266 281]
    e1xy = eigenvectors[0,:] * eigenvalues[0, 0]  # 第一主方向轴
    e2xy = eigenvectors[1,:] * eigenvalues[1, 0]  # 第二主方向轴
    p1 = (center + 0.01*e1xy).astype(np.int)  # P1:[149 403]
    p2 = (center + 0.01*e2xy).astype(np.int)  # P2:[320 332]
    theta = np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) * 180/np.pi  # 第一主方向角度 133.6

    rgb_copy = rgb.copy()
    cv2.circle(rgb_copy, center, 6, 255, -1)  # 在PCA中心位置画一个圆圈  RGB
    cv2.arrowedLine(rgb_copy, center, p1, (255, 0, 0), thickness=3, tipLength=0.1)  # 从 center 指向 pt1
    cv2.arrowedLine(rgb_copy, center, p2, (255, 0, 0), thickness=3, tipLength=0.2)  # 从 center 指向 pt2
    print("center:{}, P1:{}, P2:{}".format(center, p1, p2))
    plt.imshow(rgb_copy)
    plt.show()

    plane, inliers = segment_plane(pcd)
    colors = np.asarray(pcd.colors)
    colors[inliers] = np.array((1,0,0))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
