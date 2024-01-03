import numpy as np
import open3d as o3d

from src.graspnet_utils.graspnet_utils import (
    GraspNetUtils,
    ANN_IDS,
    SCENE_ID,
    mask_grasp2graspnet_grasp,
)
from src.geometry import segment_plane, get_plane_frame_trans
from src.mask_grasp import gen_grasp_mono, gen_grasp_depth

if __name__ == "__main__":
    # load data
    graspnet_utils = GraspNetUtils()
    obj_ids = graspnet_utils.load_object_id_list(SCENE_ID)
    for obj_id_id in range(len(obj_ids)):
        # load data from graspnet dataset
        select_obj_id = obj_ids[obj_id_id]
        obj_mask = graspnet_utils.load_obj_mask(select_obj_id, SCENE_ID, ANN_IDS[1])
        all_data = graspnet_utils.load_all_data(SCENE_ID, ANN_IDS[1])
        pcd = all_data["pcd"]
        rgb = all_data["rgb"]
        depth = all_data["depth"].astype(np.float32) / 1000
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

        # generate grasp from depth map
        grasp_trans_p_depth = gen_grasp_depth(obj_mask, depth, camK, plane)
        grasp_frame_depth = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        grasp_frame_depth.transform(grasp_trans_p_depth)

        # graspnet grasp vis
        grasp_depth = mask_grasp2graspnet_grasp(grasp_trans_p_depth)
        o3d_grasp_depth = grasp_depth.to_open3d_geometry()

        # generate grasp from monocular
        grasp_trans_p_mono = gen_grasp_mono(obj_mask, camK, plane)
        grasp_frame_mono = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        grasp_frame_mono.transform(grasp_trans_p_mono)

        # graspnet grasp vis
        grasp_mono = mask_grasp2graspnet_grasp(grasp_trans_p_mono)
        o3d_grasp_mono = grasp_mono.to_open3d_geometry().paint_uniform_color([0, 0, 1])

        o3d.visualization.draw_geometries([pcd, frame, o3d_grasp_depth, o3d_grasp_mono])
