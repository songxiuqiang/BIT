"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import subprocess
import threading

import sys
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
pp_dir = os.path.dirname(parent_dir)
sys.path.append(pp_dir)

import soft_renderer.functional as srf
import soft_renderer as sr
import utils as ut
import communication
import sam_utils as su
import cv2



from segment_anything import sam_model_registry, SamPredictor
from utils import load_config, pjoin


current_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(current_dir, '../data')

import random  
def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    print("shpe anns", type(anns))

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_sam_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)*255, [1]])
        img[m] = color_mask
    img = img[:,:,:3]  
    img = img.astype(np.uint8)  
    # cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path, normalization=False)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        
        print('vertices shape', self.template_mesh.vertices.shape)	# [1 1032 3]
        print('faces shape', self.template_mesh.faces.shape)	# [1 2600 3]
       
    
    def resetMeshFromObj(self, template_path):
        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 1.0)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
    
    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid
        
        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss
     
    def get_face_normals(self, vertices, faces):
        face_normals = []
        for batch_idx in range(faces.shape[0]):
            batch_faces = faces[batch_idx]
            batch_vertices = vertices[batch_idx]
    
            v0 = batch_vertices[batch_faces[:, 0]]
            v1 = batch_vertices[batch_faces[:, 1]]
            v2 = batch_vertices[batch_faces[:, 2]]
    
            normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
            normals = normals / torch.norm(normals, dim=-1, keepdim=True)
            
            face_normals.append(normals)
        
        face_normals = torch.stack(face_normals)
        return face_normals
    
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])  # ndimension = 3
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6

    with open('./loss_iou2.txt', 'a') as f:
         f.write(f"{predict.sum()} '\t' {target.sum()} '\t'  {intersect.sum()} {union.sum()} '\t'  {1. - (intersect / union).sum() / intersect.nelement()}\n")

    return 1. - (intersect / union).sum() / intersect.nelement()

def save_obj_file(vertices, faces, file_path):
    with open(file_path, 'w') as f:
        f.write(f"#coarse_50.obj\n# \n \n")
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()


    
def main():

    set_seed(42) 

	# RBOT dataset
    # summer_config_file = "./config/rbot/a_regular_ape.yml"
	
    # MOPED dataset
    summer_config_file = pjoin("./config/", sys.argv[1], sys.argv[2]) # 0-4
    # summer_config_file = "./config/moped/black_drill/evaluation/00.yml" # 0-4
    # summer_config_file = "./config/moped/duplo_dude/evaluation/00.yml" # 0-6
    # summer_config_file = "./config/moped/rinse_aid/evaluation/00.yml" # 0-5
    # summer_config_file = "./config/moped/toy_plane/evaluation/00.yml" # 0-4
    # summer_config_file = "./config/moped/vim_mug/evaluation/00.yml" # 0-4

    configs = su.LoadConfigSafety(config_file = summer_config_file)
    print("output_dir: ", configs['output_dir'])
    os.makedirs(configs['output_dir'], exist_ok=True)
    

    prepare_start_time = time.time()  
    # differentiable renderer
    from_sphere = False	 # Set this to False and consider the original sphere is not a sphere, so that simplify the code logic
    render_size = configs['renderer_img_size']
    img_nums = su.GenerateArray(first_num=configs['reference_imgs_nums'], second_num=configs['estimated_imgs_nums'], len =100) 
    cur_model_deformed_nums = 0
    max_model_deformed_nums = configs['max_model_deformed_nums'] 
    iteration_nums = configs['iteration_nums_in_each_deform']
    
    batch_size     = configs['batch_size']
    transform      = sr.LookAt(viewing_angle=configs['viewing_angle'])
    lighting       = sr.Lighting()
    rasterizer     = sr.SoftRasterizer(image_size=render_size, sigma_val=1e-4, aggr_func_rgb='hard',near=0.1,far=1000)
    model          = Model(configs['template_mesh']).cuda()
    img_h = configs['height']
    img_w = configs['width']

    # SAM
    sam_checkpoint = pjoin('checkpoints', configs['sam']['checkpoint'])
    model_type     = configs['sam']['model']
    device         = configs['sam']['device']
    sam            = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor      = SamPredictor(sam)

    
    save_obj_path = configs['save_obj_path']
    save_index    = configs['save_index']

	# cpp
    lsc = communication.LocalStorageCommunication(root = configs['communication']['root'], max_count = img_nums[0], from_sphere = from_sphere)
    # tracker and part of scale
    # here call the eg_BIT
    process = subprocess.Popen([configs['tracker'], summer_config_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    loop_nums = 0
        
    while(True):
        status = lsc.Scan(sleep_time = 2) # We sleep here to give the scanner time to scan the local folder, so that the C++ and Python can communicate with each other.
        lsc.set_status(status)
        
        dataGroups = lsc.parse_data()
        
        dataGroups.print_info("datgroups scaning")
        lsc.set_base_datagroups(dataGroups=dataGroups)
        if len(dataGroups.rgbs_) == lsc.max_count():
            lsc.add_new_datagroups_2_base(dataGroups=dataGroups)
            dataGroups = lsc.get_dataGroups()
            dataGroups.print_info()

            segmented_masks = su.segment(predictor=predictor, img_width=img_w, img_height=img_h, dataGroups=dataGroups, configs=configs, save_index=save_index)
            lsc.set_max_count(max_count = img_nums[cur_model_deformed_nums+1])
            segmented_masks = ut.FlipMatImgs(segmented_masks, axis = 0)	
            probs = ut.FlipMatImgs(dataGroups.probs_, axis = 0)	
            cnfds = ut.FlipMatImgs(dataGroups.masks_, axis = 0)	
            
            h, w = segmented_masks[0].shape[-2:]
            if w > h:
                segmented_masks = ut.ExtendMatImgs(segmented_masks, top = (w - h))	
                probs = ut.ExtendMatImgs(probs, top = (w - h))	
                cnfds = ut.ExtendMatImgs(cnfds, top = (w - h))	

            probs = ut.ResizeImgs(probs, width = render_size, height = render_size)
            cnfds = ut.ResizeImgs(cnfds, width = render_size, height = render_size)
            segmented_masks = ut.ResizeImgs(segmented_masks, width = render_size, height = render_size)
            for i in range(len(segmented_masks)):
                cv2.imshow("segmented_masks", segmented_masks[i])
                cv2.waitKey(100)

            probs = su.BinaringImgs(probs, 100, 255)
			# Note in this the probs actully is the mask
            probs_npy = ut.Mats2Npy(segmented_masks.copy()).astype('float32')/255
            cnfds_npy = ut.Mats2Npy(cnfds.copy()).astype('float32')/255

            Ts  = torch.from_numpy(dataGroups.Ts_)
            Ks  = torch.from_numpy(dataGroups.Ks_).to(dtype=torch.float32)
            if cur_model_deformed_nums > 0:
                model = Model(save_obj_path).cuda()

            optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
            transform.set_img_size(max(img_h, img_w))
            transform.set_K_list(Ks)
            transform.set_T(Ts)

            loop = tqdm.tqdm(list(range(0, iteration_nums)))
            batch_size = len(probs_npy)    
            cur_model_deformed_nums += 1

            print("begin to deform model")
            for i in loop:
                images_gt = torch.from_numpy(probs_npy).cuda()
                mesh, laplacian_loss, flatten_loss = model(batch_size)     

                # render
                mesh = lighting(mesh)
                mesh = transform(mesh)
                images_pred = rasterizer(mesh)    
                loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 3]) + 0.1 * laplacian_loss
                
                loop.set_description('Loss: %.4f' % (loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()            
                intermediate_result = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
                intermediate_result_flipped = cv2.flip(intermediate_result, 0) 
                cv2.imshow("intermediate_result", intermediate_result_flipped)
                cv2.waitKey(1)
                loop_nums = loop_nums + 1
                
            save_obj_path = os.path.join(configs['output_dir'], '_realtime_' + str(cur_model_deformed_nums)+'.obj')
            print("save_obj_path****", save_obj_path)
            model(1)[0].save_obj(save_obj_path, save_texture=False)
            
            if cur_model_deformed_nums == max_model_deformed_nums:
                return

    
if __name__ == '__main__':
    main()
