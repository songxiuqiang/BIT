# coding=utf-8

#
# pip install open3d==0.10.0
#

# 可根据需求删除的lib
# time: 用来为截图命名
# os: 用来存储图片

import time
import numpy as np
import open3d as o3d
import os
import cv2
import point_cloud_utils

def PathJoin(*a):
    """
    适用于ubuntu和windows的路径拼接函数
    :param a: *a: The path waitting for join 使用示例: path=PathJoin('/data1/','001.png')q
    :return: 拼接后的路径
    """
    path = a[0]
    for i in range(len(a)):
        if i==0:
            continue
        else:
            path = os.path.join(path, a[i]).replace('\\', '/')
    return path


class PointCloudRenderer:
    def __init__(self, window_shape=(512, 512), window_pos=(100, 100)):
        """
        初始化PointCloudRender
        :param m_result_dir: 存储图片的路径
        :param window_shape: (width, height) 窗口的大小
        :param window_pos: (left, top) 窗口的初始位置
        """
        self.m_window_width, self.m_window_height = window_shape
        self.m_left, self.m_top = window_pos
        self.m_rotate_count = [0]
        self.m_rotate_value = 8.0
        self.m_result_dir = None
        self.m_saveImgNum = 0
        self.m_coordinate = False
        self.m_coordinateMesh = None
        self.SetCoordinateProperty()  # init m_coordinate

    def RotateView(self, vis):
        """
        旋转回调函数
        :param vis:  vis = o3d.visualization.Visualizer()
        :return:
        """
        ctr = vis.get_view_control()
        # ctr.set_zoom(2)
        ctr.rotate(self.m_rotate_value, 0)
        if self.m_result_dir and 0 < self.m_rotate_count[0] < self.m_saveImgNum:
            save_path = os.path.join(self.m_result_dir, f'{self.m_rotate_count[0]}.png')
            vis.capture_screen_image(save_path, False)
        self.m_rotate_count[0] += 1
        return False

    def CaptureImage(self, vis):
        img_save_path = PathJoin(self.m_result_dir, "{0}.png".format(time.asctime().replace(' ', '-').replace(':', '-')))
        image_buffer = vis.capture_screen_float_buffer()
        img_tmp = np.asarray(image_buffer)
        img = img_tmp.copy()
        img[:, :, 0] = img_tmp[:, :, 2]
        img[:, :, 2] = img_tmp[:, :, 0]

        cv2.imshow("captured image", img)
        img = img * 255.0
        cv2.waitKey(1)
        if self.m_result_dir:
            cv2.imwrite(img_save_path, img)
            print("[OK] img save to {0}".format(img_save_path))
        return False

    def OpenCoordinate(self, open):
        """
        是否开启坐标系
        :param open: True 在可视化中显示坐标系; False 在可视化中关闭坐标系
        :return:
        """
        self.m_coordinate = open

    def SetCoordinateProperty(self, scale=1.0, center=(0, 0, 0)):
        """
        调整坐标系的scale, 坐标原点位置
        x, y, z 坐标轴将分别渲染为红色, 绿色, 蓝色
        :param scale: 坐标系尺寸
        :param center: tuple(x, y, z) 坐标系原点位置
        :return:
        """
        mesh_ = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_.scale(scale, center=(0, 0, 0))
        self.m_coordinateMesh = mesh_

    def RenderPointCloud(self, name, pts, result_dir=None):
        """
        最简单的可视化函数
        :param name: 窗口名
        :param pts: (numpy.array, nx3) 点云
        :param m_result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
        :return:
        """
        self.m_result_dir = result_dir
        vis = o3d.visualization.Visualizer()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        vis.add_geometry(pcd)

        key_to_callback = {}
        key_to_callback[ord("C")] = self.CaptureImage
        o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, window_name=name,
                                                             width=self.m_window_width, height=self.m_window_height,
                                                             left=self.m_left, top=self.m_top)

    def RenderMultiPointCloud(self, name, pts_list, color_list, result_dir=None):
        """
        :param name: 窗口名
        :param pts_list: list of pointcloud (numpy.array, nx3) 存放点云的list
        :param color_list: 存放颜色的list，list长度应该与pts_list相同
        :param m_result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
        :return:
        """
        self.m_result_dir = result_dir
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=name, width=self.m_window_width, height=self.m_window_height,
                          left=self.m_left, top=self.m_top)
        assert len(pts_list) == len(color_list)
        pcds = []
        for index in range(len(pts_list)):
            pcd = o3d.geometry.PointCloud()
            pts = pts_list[index]
            color = color_list[index]
            if np.any(color > 1.0):
                color = color.astype(float)/255.0
            colors = np.tile(color, (pts.shape[0], 1))  # reshape color to (pts.shape[0], 1)
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcds.append(pcd)
            vis.add_geometry(pcd)
        key_to_callback = {}
        key_to_callback[ord("C")] = self.CaptureImage
        # 另一种可视化方法
        # o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback, window_name=name,
        #                                                      width=self.m_window_width, height=self.m_window_height,
        #                                                      left=self.m_left, top=self.m_top)
        if self.m_coordinate:
            vis.add_geometry(self.m_coordinateMesh)
        # vis.register_animation_callback(self.RotateView)
        vis.run()
        vis.destroy_window()

    def RenderMultiPtsRotation(self, name, pts_list, color_list,
                                  m_rotate_value=8.0, result_dir=None, saveImgNums=500):
        """
        旋转可视化点云, 自动截图并保存到指定目录
        :param name:
        :param pts_list: list of pointcloud (numpy.array, nx3) 点云
        :param color_list: 点云颜色 例如[np.array([255,0,0]), np.array([255,255,0])], 此参数如果不为空, 长度必须和pts_list一致
        :param m_rotate_value: 该值的大小控制每次旋转的幅度 默认8.0
        :param angle_offset: 调整点云绕xyz轴的初始旋转。(值的范围为-2~2,对应-2*PI~2*PI) (numpy.float)
        :param m_result_dir: 存储图片的路径, 如果不提供则不会保存 例如: "/data/cat"
        :return:
        """
        self.m_rotate_value = m_rotate_value
        self.m_result_dir = result_dir
        self.m_saveImgNum = saveImgNums
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=name, width=512, height=512, left=50, top=25)
        assert len(pts_list) == len(color_list)

        pcds = []
        for index in range(len(pts_list)):
            pcd = o3d.geometry.PointCloud()
            pts = pts_list[index]
            color = color_list[index]
            if len(color.shape) == 1 or color.shape[0] != pts.shape[0]:
                if np.any(color > 1.0):
                    color = color.astype(float) / 255
                color = np.tile(color, (pts.shape[0], 1))  # 将color扩大为 (pts.shape[0], 1)
            else:
                print("Error in viz_pts_with_rotation: color shape {0} wrong".format(color.shape))
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(color)
            # 调整pts角度
            angle_offset = (0, 0, 0)
            x_r, y_r, z_r = angle_offset
            R = pcd.get_rotation_matrix_from_xyz((x_r * np.pi, y_r, z_r * np.pi))
            pcd = pcd.rotate(R, center=(0, 0, 0))

            pcds.append(pcd)
            vis.add_geometry(pcd)

        # o3d.visualization.draw_geometries_with_animation_callback(pcds, self.RotateView, window_name=name,
        #                                                      width=self.m_window_width, height=self.m_window_height,
        #                                                      left=self.m_left, top=self.m_top)
        if self.m_coordinate:
            vis.add_geometry(self.m_coordinateMesh)
        vis.register_animation_callback(self.RotateView)
        vis.run()
        vis.destroy_window()
        self.m_rotate_count[0] = 0


if __name__ == '__main__':
    # example of usage
    pts1 = np.random.randn(1024, 3)
    pts2 = np.random.randn(1024, 3)
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])
    #
    pcr = PointCloudRenderer()
    # pcr.RenderMultiPointCloud("multi", [pts1, pts2], [red, green], "./results/")
    # pcr.RenderPointCloud("simple", pts1, "./results/")
    # pcr.OpenCoordinate(True)
    # pcr.RenderMultiPtsRotation("multi-rotate", [pts1, pts2], [red, green], result_dir="./results/")

    modelVertices, _ = point_cloud_utils.LoadObj("../../data/RBOT/ape/ape.obj")
    pcr.RenderPointCloud("simple", modelVertices, "./results/")
