import numpy as np
import cv2


def points_pca(xys: np.array) -> list:
    """points to 2d principal directions

    Args:
        xys (np.array): points

    Returns:
        list: center, main direction vector and sub direction vector
    """
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(xys, np.array([]))
    print("mean:{}, eigenvalues:{}".format(mean.round(1), eigenvalues[:, 0].round(2)))
    center = mean[0, :]  # 近似作为目标的中心 [266 281]
    e1xy = eigenvectors[0, :] * eigenvalues[0, 0]  # 第一主方向轴
    e2xy = eigenvectors[1, :] * eigenvalues[1, 0]  # 第二主方向轴
    return center, e1xy, e2xy


def uvs_from_mask(mask: np.array) -> np.array:
    """generate uvs from mask

    Args:
        mask (np.array): mask

    Returns:
        np.array: [n, 2], uvs
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )  # OpenCV4~
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]
    ptsXY = np.squeeze(cnt).astype(np.float64)
    return ptsXY


def mask_pca(mask: np.array) -> list:
    """calculate 2d principal directions

    Args:
        mask (np.array): mask one the rgb image

    Returns:
        list: center, main direction vector and sub direction vector
    """
    ptsXY = uvs_from_mask(mask)
    return points_pca(ptsXY)
