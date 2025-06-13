import numpy as np
import os
import time

from PIL import Image

import cv2
import yaml

class FileScanner:
	def __init__(self, path, interval) -> None:
		self.path_ = path
		self.interval_ = interval
		self.files_ = set()
		self.new_files_ = set()
        
	def Scan(self):
		new_files = set(os.listdir(self.path_)) - self.files_ 
		if new_files:
			print("New files detected:")
		for file in new_files:
			print(os.path.join(self.path_, file))
		self.files_ |= new_files 
		return new_files
		

def SaveImage(image, save_path):
	image_copy = image*255
	image_copy = image_copy.astype(np.uint8)
	image_save = Image.fromarray(image_copy)
	image_save.save(save_path)

def CropImg(img,row,col,width,height):
	crop=img[row:row+height, col:col+width]
	return crop

def CenterCrop(img, target_rows, target_cols):
	row_begin = int( (img.shape[0] - target_rows) / int(2) )
	col_begin = int( (img.shape[1] - target_cols) / int(2) )
	crop = img[row_begin:row_begin+target_rows, col_begin:col_begin+target_cols,:]
	return crop

def BottomCrop(img, target_rows, target_cols):
	row_begin = 0
	col_begin = int( (img.shape[1] - target_cols) / int(2) )
	crop = img[row_begin:row_begin+target_rows, col_begin:col_begin+target_cols,:]
	return crop

def TopCrop(img, target_rows, target_cols):
	row_begin = int(img.shape[1] - target_cols)
	col_begin = int( (img.shape[1] - target_cols) / int(2) )
	crop = img[row_begin:row_begin+target_rows, col_begin:col_begin+target_cols,:]
	return crop

def ImgExtent(img, left = 0, right = 0, top =0, bottom = 0):
    
	padding = (left, top, right, bottom)
	ext = cv2.copyMakeBorder(img,left = left,right = right, top = top, bottom = bottom, borderType = cv2.BORDER_CONSTANT,value=0)
	return ext


def ColorInverse(img):
	imgInverse = 255 - img
	return imgInverse

def SaveImgAsNpy(img) -> None:
	img_size = img.shape
	print('img size',img_size)
	rows = img_size[0]
	cols = img_size[1]
	img_save_sequence = np.empty([1,4,rows,cols])
	camera_save_sequence = np.empty([1,3])

	img_save = np.empty([4,rows,cols])

	img_save[0] = img[:,:,0]
	img_save[1] = img[:,:,0]
	img_save[2] = img[:,:,0]
	img_save[3] = img[:,:,0]
	img_save_sequence[0]=img_save
	np.save('./data/rmt/coarse_img.npy', img_save_sequence)

	camera_save = [2.73, 1.77, 30]
	camera_save_sequence[0] = camera_save
	np.save('./data/rmt/coarse_camera.npy', camera_save_sequence)

def ResizeK(K, scaled = 1.0):
    K[0,0] *= scaled
    K[0,2] *= scaled
    K[1,1] *= scaled
    K[1,2] *= scaled

def ResizeImgs(imgs, width, height):
	scaled_imgs = []
	for img in imgs:
		scaled_imgs.append(cv2.resize(img, (width, height)))
	return scaled_imgs

def BlurImgs(imgs, blur_size = 5):
	res = []
	for img in imgs:
		res.append(cv2.blur(img, (blur_size, blur_size)))
	return res


def LoadImgsAsMat(path, nums = 3, stride = 1, inverseLoad = False):
	ips = []	# img paths
	imgs = [] 	# result 
	for filename in os.listdir(path):
		file_path = os.path.join(path, filename)
		ips.append(file_path)
  
	ips = sorted(ips, key=lambda x: (len(x), x))
	
	load_id = 0
	for ip in ips:
		if load_id > (nums - 1) * stride:
			break
		if load_id % stride != 0: 
			load_id = load_id + 1
			continue			
		img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
		imgs.append(img)
		load_id += 1
	return imgs

def FlipMatImgs(imgs, axis = 0):
	if axis == -1:
		return imgs
	fliped_imgs = []
	for img in imgs:
		flip = np.flip(img, axis = axis)
		fliped_imgs.append(flip)
	return fliped_imgs

def ExtendMatImgs(imgs, top = 0, left = 0, right = 0, bottom = 0):
	ext_imgs = []
	for img in imgs:
		print("img size", img.shape)
		ext_img = ImgExtent(img, top = top, left = left, right = right, bottom = bottom)
		ext_imgs.append(ext_img)
	return ext_imgs

def Mats2Npy(imgs):
	img_npys = []
	for img in imgs:
		img_array = np.tile(img,(4,1,1))
		img_npys.append(img_array)
		# cv2.imshow("imgs",img)
		cv2.waitKey(10)

	img_npys = np.stack(img_npys, axis = 0)
	print('img_npys.shape',img_npys.shape)
	return img_npys


def LoadRmtImgsAsNpy(path, nums = 3, stride = 1):
	ips = []
	for filename in os.listdir(path):
		file_path = os.path.join(path, filename)
		ips.append(file_path)
  
	ips = sorted(ips, key=lambda x: (len(x), x))

	img_npys = []
	load_id = 0
	for ip in ips:
		if load_id >= nums:
			break
		if load_id % stride != 0:
			load_id = load_id + 1
			continue			
		img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
		img = np.flip(img, axis = 0)
		img = ImgExtent(img, top = 128, left = 0, right = 0, bottom = 0)
		img_array = np.tile(img,(4,1,1))
		img_npys.append(img_array)
		print('img_array.shape',img_array.shape)
		load_id = load_id +1

	img_npys = np.stack(img_npys, axis = 0)
	print('img_npys.shape',img_npys.shape)
	return img_npys

def LoadImgsAsNpy(path, nums = 3, stride = 1):
	ips = []
	for filename in os.listdir(path):
		file_path = os.path.join(path, filename)
		ips.append(file_path)
  
	ips = sorted(ips, key=lambda x: (len(x), x))

	img = cv2.imread(ips[0], cv2.IMREAD_GRAYSCALE)
	rows, cols = img.shape[0], img.shape[1] # 640
	img_size = max(cols, rows)
	top, right = (int)((img_size-rows)/2), (int)((img_size-cols)/2)
	img_npys = []
	load_id = 0
	for ip in ips:
		# print('ip', ip)
		if load_id >= nums:
			break
		if load_id % stride != 0:
			load_id = load_id + 1
			continue			
		img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
		img_ext = cv2.copyMakeBorder(img, top, top, right, right, cv2.BORDER_CONSTANT, 0)
		img_array = np.tile(img_ext,(4,1,1))
		img_npys.append(img_array)
		print('img_array.shape',img_array.shape)
		cv2.imshow("img_ext",img_ext)
		cv2.waitKey(0)
		load_id = load_id +1

	img_npys = np.stack(img_npys, axis = 0)
	print('img_npys.shape',img_npys.shape)
	return img_npys

def VisualNpy(npy_array, waitKey = 0,win_name = 'npy'):
	r = npy_array[0]
	g = npy_array[1]
	b = npy_array[2]
	a = npy_array[3]

	rgba = np.stack([r, g, b, a], axis=-1)
	bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
	bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
	cv2.imshow(win_name, bgr)
	cv2.waitKey(waitKey)

def read_obj(filename):
    vertices = []
    texcoords = []
    normals = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                vertices.append(list(map(float, values[1:4])))
                if len(values) >= 7:
                    texcoords.append(list(map(float, values[4:7])))
                if len(values) >= 10:
                    normals.append(list(map(float, values[7:10])))
            elif values[0] == 'f':
                face = []
                texcoord = []
                normal = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0])-1)
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoord.append(int(w[1])-1)
                    else:
                        texcoord.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        normal.append(int(w[2])-1)
                    else:
                        normal.append(0)
                faces.append(face)
    vertices = np.array(vertices)
    texcoords = np.array(texcoords)
    normals = np.array(normals)
    faces = np.array(faces)

    return vertices, faces, normals, texcoords

def write_obj(filename, vertices, faces, normals=None, texcoords=None):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v %f %f %f\n' % tuple(v))
        if texcoords is not None:
            for vt in texcoords:
                f.write('vt %f %f\n' % tuple(vt))
        if normals is not None:
            for vn in normals:
                f.write('vn %f %f %f\n' % tuple(vn))
        for face in faces:
            if normals is not None and texcoords is not None:
                f.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                    face[0]+1, face[0]+1, face[0]+1,
                    face[1]+1, face[1]+1, face[1]+1,
                    face[2]+1, face[2]+1, face[2]+1))
            elif normals is not None:
                f.write('f %d//%d %d//%d %d//%d\n' % (
                    face[0]+1, face[0]+1,
                    face[1]+1, face[1]+1,
                    face[2]+1, face[2]+1))
            elif texcoords is not None:
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0]+1, face[0]+1,
                    face[1]+1, face[1]+1,
                    face[2]+1, face[2]+1))
            else:
                f.write('f %d %d %d\n' % (
                    face[0]+1, face[1]+1, face[2]+1))

def scale_obj(filename, save_path, scale_factor):
    vertices, faces, normals, texcoords = read_obj(filename=filename)
    vertices *= scale_factor
    if texcoords is not None:
        texcoords *= scale_factor
    if normals is not None:
        normals /= scale_factor
    write_obj(save_path, vertices, faces, normals, texcoords)

from scipy.spatial.transform import Rotation

import numpy as np

def calculate_rotation_angle(r1, r2):
    R1 = r1.as_matrix()
    R2 = r2.as_matrix()
    R = np.dot(R1.T, R2)
    trace = np.trace(R)
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    angle_deg = np.degrees(angle)

    return angle_deg


def find_uniform_poses(poses, n, min_rotation):
    res = []
    rotations = [pose[:3, :3] for pose in poses]
    
    num_poses = len(rotations)
    indices = list(range(num_poses))

    rotation_angles = np.zeros((num_poses, num_poses))
    for i in range(num_poses):
        for j in range(i+1, num_poses):
            r1 = Rotation.from_matrix(rotations[i])
            r2 = Rotation.from_matrix(rotations[j])
            rotation_angles[i, j] = calculate_rotation_angle(r1, r2)
    
    selected_indices = []
    selected_indices.append(indices.pop(0))

    for i in range(num_poses-1):
        angle = min(rotation_angles[i, j] for j in range(i+1, num_poses))
        if angle >= min_rotation:
            selected_indices.append(i)
        if len(selected_indices) == n:
            break   
    
    for idx in selected_indices:
        print('idx', idx)
        res.append(poses[idx])
    res = np.stack(res, axis = 0)
    return res, selected_indices


def load_indexed_imgs_as_mat(path, indices):
	ips = []	# img paths
	imgs = [] 	# result 
	for filename in os.listdir(path):
		file_path = os.path.join(path, filename)
		ips.append(file_path)
  
	ips = sorted(ips, key=lambda x: (len(x), x))
	
	for index in indices:
		img = cv2.imread(ips[index], cv2.IMREAD_GRAYSCALE)
		imgs.append(img)
	return imgs

def load_K(posefile, from_rows = 0, from_cols = 1, split_charactor = ' '):
	with open(posefile, 'r') as f:
		content = f.readlines()
  
	data_list = []
	K_list = []

	for line in content:
		if from_rows:
			from_rows = from_rows -1
			continue
		line_data = line.strip().split(split_charactor)
		line_data = [float(x) for x in line_data]
		line_data = line_data[0 + from_cols:]
		K_list.append(line_data)
	K_list = np.stack(K_list, axis = 0)
	return K_list
 
 
def load_poses(posefile, from_rows = 0, from_cols = 1, split_charactor = ' ', centered = False, scale = 1.0):
    with open(posefile, 'r') as f:
        content = f.readlines()
    
    data_list = []
    T_list = []
    
    for line in content:
        if from_rows:
            from_rows = from_rows -1
            continue
        line_data = line.strip().split(split_charactor)
        line_data = [float(x) for x in line_data]
        line_data = line_data[0 + from_cols:]
        data_list.append(line_data)
        R = np.array(line_data[:9], dtype = np.float32).reshape((3,3))
        t = np.array(line_data[9:], dtype = np.float32).reshape((3,1))
        if centered:
            t[0], t[1] = 0, 0
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = R
        T[:3,3] = scale * t.flatten()
        T_list.append(T)
    T_list = np.stack(T_list, axis = 0)
    return T_list

def load_poses_realtime(dir_path, from_rows = 0, from_cols = 1, split_charactor = ' ', scale = 1.0, nums = 10):
    
    poses = []
    data = []
    posesFiles = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        posesFiles.append(file_path) 
    posesFiles = sorted(posesFiles, key=lambda x: (len(x), x))
    if len(posesFiles) > 10:
        posesFiles = posesFiles[len(posesFiles)-nums:]
    
    for poseFile in posesFiles:
        with open(poseFile, 'r') as f:
            content = f.readlines()
    
        for line in content:
            if from_rows:
                from_rows = from_rows -1
                continue
            line_data = line.strip().split(split_charactor)
            line_data = [float(x) for x in line_data]
            line_data = line_data[0 + from_cols:]
            R = np.array(line_data[:9], dtype = np.float32).reshape((3,3))
            t = np.array(line_data[9:], dtype = np.float32).reshape((3,1))
            T = np.eye(4, dtype=np.float32)
            T[:3,:3] = R
            T[:3,3] = scale * t.flatten()
            poses.append(T)
    poses = np.stack(poses, axis = 0)
    return poses

def compute_azimuth_elevation(R):
    v = R[:, 2]  
    theta = np.arccos(v.dot(np.array([0, 0, 1]))) 
    if np.isclose(theta, 0.0):
        phi = 0.0 
    else:
        if R[1, 0] >= 0:
            phi = np.arccos(R[0, 0] / np.cos(theta))
        else:
            phi = 2 * np.pi - np.arccos(R[0, 0] / np.cos(theta))
        print(R[0, 0],  np.cos(theta) ,np.arccos(R[0, 0] / np.cos(theta)))
    return phi, theta


def compare_rendered_imgs(dir_1, dir_2):
	ips_1 = []
	ips_2 = []
	for filename in os.listdir(dir_1):
		file_path = os.path.join(dir_1, filename)
		ips_1.append(file_path)
	for filename in os.listdir(dir_2):
		file_path = os.path.join(dir_2, filename)
		ips_2.append(file_path)
  
	ips_1 = sorted(ips_1, key=lambda x: (len(x), x))
	ips_2 = sorted(ips_2, key=lambda x: (len(x), x))

	for i in range(0,len(ips_1)):
		print(ips_1[i])
		img_1 = cv2.imread(ips_1[i])
		img_2 = cv2.imread(ips_2[i])
		img_compare=cv2.addWeighted(img_1,0.5,img_2,0.5,0)
		cv2.imshow('img compare',img_compare)
		cv2.waitKey(0)
            
def load_config(config_path="config/dataset.yml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def pjoin(*args):
    """Join one or more path components intelligently."""
    return os.path.join(*args)