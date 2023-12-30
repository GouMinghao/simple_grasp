from graspnetAPI import GraspNet, Grasp
import os
import numpy as np

GRASPNET_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "graspnet_sample"
)
SCENE_ID = 0
ANN_IDS = [16, 135]
DEFAULT_CAMERA = "realsense"

# grasp parameters (https://graspnetapi.readthedocs.io/en/latest/grasp_format.html)
GRASP_SCORE = 1
GRASP_WIDTH = 0.1  # 0.1meter
GRASP_HEIGHT = 0.02
GRASP_DEPTH = 0.03  # total 0.05m
GRASP_OBJECT_ID = 0


class GraspNetUtils:
    def __init__(self, root=GRASPNET_ROOT, camera=DEFAULT_CAMERA):
        self.root = root
        self.camera = camera
        self.gn = GraspNet(
            root=self.root, camera=self.camera, split="custom", sceneIds=[0]
        )

    def load_object_id_list(self, scene_id):
        obj_ids = self.gn.getObjIds(scene_id)
        return obj_ids

    def load_all_data(self, scene_id=SCENE_ID, ann_id=ANN_IDS[0]):
        rgb = self.gn.loadRGB(sceneId=scene_id, camera=self.camera, annId=ann_id)
        depth = self.gn.loadDepth(sceneId=scene_id, camera=self.camera, annId=ann_id)
        all_mask = self.gn.loadMask(sceneId=scene_id, camera=self.camera, annId=ann_id)
        pcd = self.gn.loadScenePointCloud(scene_id, self.camera, ann_id)
        return {"rgb": rgb, "depth": depth, "all_mask": all_mask, "pcd": pcd}

    def load_obj_mask(self, obj_id, scene_id=SCENE_ID, ann_id=ANN_IDS[0]):
        all_mask = self.gn.loadMask(sceneId=scene_id, camera=self.camera, annId=ann_id)
        obj_mask = (all_mask == (obj_id + 1)).astype(np.uint8)
        return obj_mask

    def load_camK(self, camera=DEFAULT_CAMERA):
        return np.load(
            os.path.join(self.root, "scenes", "scene_0000", camera, "camK.npy")
        )


def mask_grasp2graspnet_grasp(grasp_trans: np.array) -> Grasp:
    """convert mask grasp format to graspnet Grasp format

    Args:
        grasp_trans (np.array): [4, 4] homogeneous matrix

    Returns:
        Grasp: graspnet Grasp.
    """
    graspnet_grasp_trans_p = grasp_trans.copy()
    graspnet_grasp_trans_p[2, 3] += GRASP_DEPTH
    graspnet_grasp_trans_p[:3, :3] = graspnet_grasp_trans_p[:3, [2, 1, 0]]
    graspnet_grasp_trans_p[:3, 2] *= -1
    grasp = Grasp(
        GRASP_SCORE,
        GRASP_WIDTH,
        GRASP_HEIGHT,
        GRASP_DEPTH,
        graspnet_grasp_trans_p[:3, :3],
        graspnet_grasp_trans_p[:3, 3],
        GRASP_OBJECT_ID,
    )
    return grasp
