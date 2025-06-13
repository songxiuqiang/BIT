# This file implements the communication betwwen C++ and Python

import numpy as np
import time
import os
import copy
import cv2
from utils import pjoin
class DataGroups:
    def __init__(self, rgbs=None, probs=None, masks=None, Ts=None, Ks=None, origin_rois=None, target_rois=None, target_sizes=None):
        self.rgbs_ = rgbs if rgbs is not None else []
        self.probs_ = probs if probs is not None else []
        self.masks_ = masks if masks is not None else []
        self.Ts_ = Ts if Ts is not None else []
        self.Ks_ = Ks if Ks is not None else []
        self.origin_rois_ = origin_rois if origin_rois is not None else []
        self.target_rois_ = target_rois if target_rois is not None else []
        self.target_sizes_ = target_sizes if target_sizes is not None else []
        
    def add(self, other):
        for rgb in other.rgbs_:
            self.rgbs_.append(rgb)
        for prob in other.probs_:
            self.probs_.append(prob)
        for mask in other.masks_:
            self.masks_.append(mask)
        self.Ts_ = np.vstack((self.Ts_, other.Ts_))
        self.Ks_ = np.vstack((self.Ks_, other.Ks_))
        self.origin_rois_ = np.vstack((self.origin_rois_, other.origin_rois_))
        self.target_rois_ = np.vstack((self.target_rois_, other.target_rois_))
        self.target_sizes_ = np.vstack((self.target_sizes_, other.target_sizes_))

    def print_info(self, title = "") -> None:
        print(title)
        headers = ["rgbs_", "probs_", "Ts_", "Ks_", "origin_rois_", "target_rois_", "target_sizes_"]
        counts = [len(self.rgbs_), len(self.probs_), len(self.Ts_), len(self.Ks_), len(self.origin_rois_), len(self.target_rois_), len(self.target_sizes_)]
        # Determine the column width
        col_width = max(len(header) for header in headers) + 2  # Adding space for padding
        # Print the headers
        header_line = ''.join(header.ljust(col_width) for header in headers)
        print(header_line)
        # Print the counts
        count_line = ''.join(str(count).ljust(col_width) for count in counts)
        print(count_line)
        
    def show_imgs(self, type = "probs", wait_key = 0):
        if type == "probs":
            for prob in self.probs_:
                # cv2.imshow("probs in dataGroups",prob)
                cv2.waitKey(wait_key)
        if type == "rgbs":
            for rgb in self.rgbs_:
                # cv2.imshow("rgbs in dataGroups", rgb)
                cv2.waitKey(wait_key)
        if type == "masks":
            for mask in self.masks_:
                cv2.imshow("mask in dataGroups", mask)
                cv2.waitKey(wait_key)
        
        
    def __repr__(self):
        return (f"DataGroups(rgbs={len(self.rgbs_)} Tensors, "
                f"probs={len(self.probs_)} Tensors, masks={len(self.masks_)} Tensors, "
                f"Ts={len(self.Ts_)} Tensors, Ks={len(self.Ks_)} Tensors, "
                f"origin_rois={len(self.origin_rois_)} items, target_rois={len(self.target_rois_)} items, "
                f"target_sizes={len(self.target_sizes_)} items)")
        
class LocalStorageCommunication:
	def __init__(self, root, rgb_folder = "img", prob_folder = "prob", mask_folder = "mask", \
                       poses_folder = "pose", K_folder = "K", \
                       origin_roi_folder  = "originRoi", target_roi_folder = "targetRoi", target_size_folder = "targetSize", \
                       max_count = 30, from_sphere = False) -> None:
		self.root_ = root
        
		self.dataGroups_ = DataGroups()
		self.baseDataGroups_ = DataGroups()
  
		self.status_ = False
  
        # RGB 
		self.rgb_root_ = pjoin(self.root_, rgb_folder)
		self.rgb_files_ = list()
		self.rgbs_ = list()

        # probability map 
		self.prob_root_ = pjoin(self.root_, prob_folder)
		self.prob_files_ = list()
		self.probs_ = list()

        # mask
		self.mask_root_ = pjoin(self.root_, mask_folder)
		self.mask_files_ = list()
		self.masks_ = list()
  
        # pose T
		self.pose_root_ = pjoin(self.root_, poses_folder)
		self.pose_files_ = list()
		self.poses_ = list()

        # 相机内参K(可变)
		self.K_root_ = pjoin(self.root_, K_folder)
		self.K_files_ = list()
		self.Ks_ = list()

		# origin roi
		self.origin_roi_root_ = pjoin(self.root_, origin_roi_folder)
		self.origin_roi_files_ = list()
		self.origin_rois_ = list()

		# target roi 
		self.target_roi_root_ = pjoin(self.root_, target_roi_folder)
		self.target_roi_files_ = list()
		self.target_rois_ = list()

		# target size
		self.target_size_root_ = pjoin(self.root_, target_size_folder)
		self.target_size_files_ = list()
		self.target_sizes_ = list()
        
        # 模型
		self.models_files_ = list()

		self.base_data_status_ = True
		self.max_count_ = max_count
		self.curr_count_ = 0
        
		if from_sphere == True:
			self.max_count_ = 1

	def Scan(self, sleep_time = 0):
        
		if sleep_time != 0:
			time.sleep(sleep_time)
  
		print(self.rgb_root_)
		# RGB 
		new_rgb_files = [item for item in os.listdir(self.rgb_root_) if item not in self.rgb_files_]
		new_rgb_files.sort()
		self.rgb_files_ += new_rgb_files

        # prob
		new_prob_files = [item for item in os.listdir(self.prob_root_) if item not in self.prob_files_]
		new_prob_files.sort()
		self.prob_files_ += new_prob_files
        
        # mask 
		new_mask_files = [item for item in os.listdir(self.mask_root_) if item not in self.mask_files_]
		new_mask_files.sort()
		self.mask_files_ += new_mask_files
        
		new_pose_files = [item for item in os.listdir(self.pose_root_) if item not in self.pose_files_]
		new_pose_files.sort()
		self.pose_files_ += new_pose_files

		new_K_files = [item for item in os.listdir(self.K_root_) if item not in self.K_files_]
		new_K_files.sort()
		self.K_files_ += new_K_files

		new_origin_roi_files = [item for item in os.listdir(self.origin_roi_root_) if item not in self.origin_roi_files_]
		new_origin_roi_files.sort()
		self.origin_roi_files_ += new_origin_roi_files

		new_target_roi_files = [item for item in os.listdir(self.target_roi_root_) if item not in self.target_roi_files_]
		new_target_roi_files.sort()
		self.target_roi_files_ += new_target_roi_files

		new_target_size_files = [item for item in os.listdir(self.target_size_root_) if item not in self.target_size_files_]
		new_target_size_files.sort()
		self.target_size_files_ += new_target_size_files

		self.curr_count_ = len(self.K_files_)
		return (self.curr_count_ >= self.max_count_)

	def set_status(self, status):
		self.status_ = status
  
	def parse_data(self):
		if not self.status_:
			print("Data from cpp not enough, waiting for more ...")
			return DataGroups()

		imgs = []
		probs = []
		masks = []
		poses = []
		Ks = []
		origin_rois = []
		target_rois = []
		target_sizes = []

		# select the latest max_count_ data
		begin = self.curr_count_ - self.max_count_

		# RGB
		newest_rgb_files = self.rgb_files_[begin:]
		for img_file in newest_rgb_files:
			imgs.append(cv2.imread(self.rgb_root_ + "/" + img_file))

		# prob
		newest_prob_files = self.prob_files_[begin:]
		for prob_file in newest_prob_files:
			probs.append(cv2.imread(self.prob_root_ + "/" + prob_file, cv2.IMREAD_GRAYSCALE))

		# mask
		newest_mask_files = self.mask_files_[begin:]
		for mask_file in newest_mask_files:
			masks.append(cv2.imread(self.mask_root_ + "/" + mask_file, cv2.IMREAD_GRAYSCALE))

		# parsing pose 
		newest_pose_files = self.pose_files_[begin:]
		for pose_file in newest_pose_files:
			with open(self.pose_root_ + "/" + pose_file, 'r') as f:
				content = f.readlines()
				for line in content:
					line_data = line.strip().split(' ')
					line_data = [float(x) for x in line_data]
					R = np.array(line_data[:9], dtype = np.float32).reshape((3,3))
					t = np.array(line_data[9:], dtype = np.float32).reshape((3,1))
					T = np.eye(4, dtype=np.float32)
					T[:3,:3] = R
					T[:3,3] = t.flatten()
					poses.append(T)
		poses = np.stack(poses, axis = 0)

		# parsing K
		newest_K_files = self.K_files_[begin:]
		for K_file in newest_K_files:
			with open(self.K_root_ + "/" + K_file, 'r') as f:
				content = f.readlines()
				for line in content:
					line_data = line.strip().split(' ')
					line_data = [float(x) for x in line_data]
					Ks.append(line_data)
		Ks = np.stack(Ks, axis = 0)

		# parsing origin roi
		newest_origin_roi_files = self.origin_roi_files_[begin:]
		for origin_roi_file in newest_origin_roi_files:
			with open(self.origin_roi_root_ + "/" + origin_roi_file, 'r') as f:
				content = f.readlines()
				for line in content:
					line_data = line.strip().split(' ')
					line_data = [int(x) for x in line_data]
					origin_rois.append(line_data)
		origin_rois = np.stack(origin_rois, axis = 0)

		# parsing target roi
		newest_target_roi_files = self.target_roi_files_[begin:]
		for target_roi_file in newest_target_roi_files:
			with open(self.target_roi_root_ + "/" + target_roi_file, 'r') as f:
				content = f.readlines()
				for line in content:
					line_data = line.strip().split(' ')
					line_data = [int(x) for x in line_data]
					target_rois.append(line_data)
		target_rois = np.stack(target_rois, axis = 0)

     	# parsing target size
		newest_target_size_files = self.target_size_files_[begin:]
		for target_size_file in newest_target_size_files:
			with open(self.target_size_root_ + "/" + target_size_file, 'r') as f:
				content = f.readlines()
				for line in content:
					line_data = line.strip().split(' ')
					line_data = [int(x) for x in line_data]
					target_sizes.append(line_data)
		target_sizes = np.stack(target_sizes, axis = 0)

		return DataGroups(rgbs=imgs, probs=probs, masks=masks, Ts=poses, Ks=Ks, origin_rois=origin_rois, target_rois=target_rois, target_sizes=target_sizes)


	def SetBaseDatas(self, imgs, probs, masks, poses, Ks, origin_rois, target_rois, target_sizes):
		if (not self.status_) or self.base_data_status_ ==  False:
			return
		
		self.rgbs_ = imgs.copy()
		self.probs_ = probs.copy()
		self.masks_ = masks.copy()
		self.poses_ = poses.copy()
		print("self.poses_ size", len(self.poses_))
		self.Ks_ = Ks.copy()
		self.origin_rois_ = origin_rois.copy()
		self.target_rois_ = target_rois.copy()
		self.target_sizes_ = target_sizes.copy()
		self.base_data_status_ = False
		return

	def set_base_datagroups(self, dataGroups):
		if self.status_ == False:
			return
		if self.base_data_status_ ==  False:
			return
		self.baseDataGroups_ = copy.deepcopy(dataGroups)
		self.base_data_status_ = False

	def MixedBaseAndParsedDatas(self, imgs, probs, masks, poses, Ks, origin_rois, target_rois, target_sizes):
		res_rgbs = imgs + self.rgbs_.copy()
		res_probs = probs + self.probs_.copy()
		res_masks = masks + self.masks_.copy()
		res_poses = np.concatenate((poses,self.poses_))
		res_Ks = np.concatenate((Ks,self.Ks_))
		res_origin_rois = np.concatenate((origin_rois, self.origin_rois_))
		res_target_rois = np.concatenate((target_rois, self.target_rois_))
		res_target_sizes = np.concatenate((target_sizes, self.target_sizes_))
		return res_rgbs, res_probs, res_masks, res_poses, res_Ks, res_origin_rois, res_target_rois, res_target_sizes

	def add_new_datagroups_2_base(self, dataGroups):
		self.dataGroups_=copy.deepcopy(self.baseDataGroups_)
		self.dataGroups_.add(dataGroups)
		print("self.poses_ size", len(self.dataGroups_.Ts_))
		return self.dataGroups_

	def get_dataGroups(self):
		return copy.deepcopy(self.dataGroups_)

	def max_count(self):
		return self.max_count_
	
	def set_max_count(self, max_count):
		self.max_count_ = max_count	