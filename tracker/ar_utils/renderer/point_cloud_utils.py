# coding=utf-8

import os
import cv2
import numpy as np


def LoadObj(path_to_file):
    """
    加载obj模型
    可将vertices作为点云来可视化
    :param path_to_file: obj模型路径
    :return: vertices, faces 顶点 和 面片
    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces


def LoadDepth(depth_path):
    """
    加载深度图
    :param depth_path: 深度图路径
    :return: depth16 深度图 np.uint16
    """
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16 == 32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def GetBbox(bbox, img_height=480, img_width=640):
    """
    给定检测的长方形bbox的两个坐标(x1, y1)和(x2, y2)， 计算出其在图像上的正方形裁剪区域，用于裁剪以及之后的卷积等操作
    :param bbox: 检测算法的结果 (y1, x1, y2, x2)
    :param img_height: 图像的高
    :param img_width: 图像的宽
    :return: rmin, rmax, cmin, cmax 边界值, 用于裁剪图像,例如: img[rmin:rmax, cmin:cmax, :]
    """
    y1, x1, y2, x2 = bbox
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax
