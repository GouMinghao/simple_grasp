import numpy as np
import open3d as o3d

from src.graspnet_utils.graspnet_utils import (
    GraspNetUtils,
    ANN_IDS,
    SCENE_ID,
    mask_grasp2graspnet_grasp,
)
from src.geometry import segment_plane, get_plane_frame_trans
from src.mask_grasp import gen_grasp_mono

if __name__ == "__main__":
    # load data
    graspnet_utils = GraspNetUtils()
    obj_ids = graspnet_utils.load_object_id_list(SCENE_ID)
    for obj_id_id in range(len(obj_ids)):
        # load data from graspnet dataset
        select_obj_id = obj_ids[obj_id_id]
        obj_mask = graspnet_utils.load_obj_mask(select_obj_id, SCENE_ID, ANN_IDS[0])
        all_data = graspnet_utils.load_all_data(SCENE_ID, ANN_IDS[0])
        pcd = all_data["pcd"]
        rgb = all_data["rgb"]
        camK = graspnet_utils.load_camK()

        # segmenta plain from point cloud (It can be given, too)
        plane, inliers = segment_plane(pcd)

        # draw colors
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
        colors = np.asarray(pcd.colors)
        colors[inliers] = np.array((1, 0, 0))
        colors /= 2
        pcd.colors = o3d.utility.Vector3dVector(colors)
        plane_trans = get_plane_frame_trans(plane)
        pcd.transform(np.linalg.inv(plane_trans))

        # generate grasp
        grasp_trans_p = gen_grasp_mono(obj_mask, camK, plane)
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        grasp_frame.transform(grasp_trans_p)

        # graspnet grasp vis
        grasp = mask_grasp2graspnet_grasp(grasp_trans_p)
        o3d_grasp = grasp.to_open3d_geometry()
        o3d.visualization.draw_geometries([pcd, frame, grasp_frame, o3d_grasp])
