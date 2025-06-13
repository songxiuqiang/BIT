
import math
import numpy as np
import torch
import torch.nn as nn

import soft_renderer as sr
import soft_renderer.functional as srf


def perspective(vertices, img_size, angle = 30., K = None, K_list = None):
    if img_size is None:
        print("img size can not be None in perspective()")
        exit()
    if K is not None:
        device = vertices.device
        width = torch.tensor(img_size,device=device)[None]
        width = width[:, None]
        height = torch.tensor(img_size,device=device)[None]
        height = height[:, None]
        fx = torch.tensor(K[0,0], device=device)[None]
        fy = torch.tensor(K[1,1], device=device)[None]
        cx = torch.tensor(K[0,2], device=device)[None]
        cy = torch.tensor(K[1,2], device=device)[None]
        fx = fx[:,None]
        fy = fy[:,None]
        cx = cx[:,None]
        cy = cy[:,None]
       
        print("vertices shape ", vertices.shape) # 20*n*3

        f = 3000
        n = 1 
        z = vertices[:, :, 2]
        x = (2*fx*vertices[:, :, 0]/width  + z*(1-2*cx/width)) / -z 
        y = (2*fy*vertices[:, :, 1]/height + z*(2*cy/height-1)) / z
        z = -z
        vertices = torch.stack((x, y, z), dim=2)
        return vertices
    elif K_list is not None:
        device = vertices.device
        width = torch.tensor(img_size,device=device)[None]
        width = width[:, None]
        height = torch.tensor(img_size,device=device)[None]
        height = height[:, None]

        if K_list.device.type != 'cuda':
            K_list = K_list.to(device)
            
        fx = K_list[:,0]
        fy = K_list[:,1]
        cx = K_list[:,2]
        cy = K_list[:,3]

        fx = fx[:, None].expand(vertices.shape[0], vertices.shape[1])
        fy = fy[:, None].expand(vertices.shape[0], vertices.shape[1])
        cx = cx[:, None].expand(vertices.shape[0], vertices.shape[1])
        cy = cy[:, None].expand(vertices.shape[0], vertices.shape[1])

        f = 3000
        n = 1 
        z = vertices[:, :, 2]
        x = (2*fx*vertices[:, :, 0]/width  + z*(1-2*cx/width)) / -z 
        y = (2*fy*vertices[:, :, 1]/height + z*(2*cy/height-1)) / z
        # z = (z*(f+n)/(n-f) - z*2*f*n/(n-f)) / -z 
        z = -z
        vertices = torch.stack((x, y, z), dim=2)
        return vertices
    else :
        print("Set perspective by angle")
        '''
        Compute perspective distortion from a given angle
        '''
        if (vertices.ndimension() != 3):
            raise ValueError('vertices Tensor should have 3 dimensions')
        device = vertices.device
        angle = torch.tensor(angle / 180 * math.pi, dtype=torch.float32, device=device)
        angle = angle[None]
        width = torch.tan(angle)
        width = width[:, None]
        z = vertices[:, :, 2]
        x = vertices[:, :, 0] / z / width
        y = vertices[:, :, 1] / z / width
        vertices = torch.stack((x, y, z), dim=2)
        return vertices


def orthogonal(vertices, scale=1.):
    '''
    Compute orthogonal projection from a given angle
    To find equivalent scale to perspective projection
    set scale = focal_pixel / object_depth  -- to 0~H/W pixel range
              = 1 / ( object_depth * tan(half_fov_angle) ) -- to -1~1 pixel range
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] * scale
    y = vertices[:, :, 1] * scale
    vertices = torch.stack((x, y, z), dim=2)
    return vertices


class Transform(nn.Module):
    def __init__(self):
        super().__init__()

    def transform(self, vertices):
        raise NotImplementedError()

    def forward(self, mesh):
        new_vertices = self.transform(mesh.vertices)
        faces = mesh.faces
        textures = mesh.textures
        texture_res = mesh.texture_res
        texture_type = mesh.texture_type
        return sr.Mesh(new_vertices, faces, textures, texture_res, texture_type)


class Projection(Transform):
    def __init__(self, P, dist_coeffs=None, orig_size=512):
        super().__init__()
        '''
        Calculate projective transformation of vertices given a projection matrix
        P: 3x4 projection matrix
        dist_coeffs: vector of distortion coefficients
        orig_size: original size of image captured by the camera
        '''

        self.P = P
        self.dist_coeffs = dist_coeffs
        self.orig_size = orig_size

        if isinstance(self.P, np.ndarray):
            self.P = torch.from_numpy(self.P).cuda()
        if self.P is None or self.P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
            raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
        if dist_coeffs is None:
            self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(self.P.shape[0], 1)

    def transform(self, vertices):
        vertices = torch.cat([vertices, torch.ones_like(vertices[:, :, None, 0])], dim=-1)
        vertices = torch.bmm(vertices, self.P.transpose(2, 1))
        x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        x_ = x / (z + 1e-5)
        y_ = y / (z + 1e-5)

        # Get distortion coefficients from vector
        k1 = self.dist_coeffs[:, None, 0]
        k2 = self.dist_coeffs[:, None, 1]
        p1 = self.dist_coeffs[:, None, 2]
        p2 = self.dist_coeffs[:, None, 3]
        k3 = self.dist_coeffs[:, None, 4]

        # we use x_ for x' and x__ for x'' etc.
        r = torch.sqrt(x_ ** 2 + y_ ** 2)
        x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
        y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 * (r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
        x__ = 2 * (x__ - self.orig_size / 2.) / self.orig_size
        y__ = 2 * (y__ - self.orig_size / 2.) / self.orig_size
        vertices = torch.stack([x__, y__, z], dim=-1)
        return vertices


class LookAt(Transform):
    def __init__(self, perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None, img_size = None):
        super(LookAt, self).__init__()
        print('*********************Init LookAt***********************')
        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye
        self._T = None
        self._K = None
        self._K_list = None
        self.img_size_ = img_size

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def set_eyes_from_angles(self, distances, elevations, azimuths):
        self._eye = srf.get_points_from_angles(distances, elevations, azimuths)
        # print('set_eyes_from_angles self._eye', self._eye.shape)

    def set_T(self, T):
        self._T = T
        # print('set_T self._T', self._T.shape)
    
    def set_K(self, K):
        self._K = K
    
    def set_img_size(self, img_size):
        self.img_size_ = img_size
        
    def set_K_list(self, K_list):
        self._K_list = K_list
            
    def set_eyes(self, eyes):
        self._eye = eyes

    @property
    def eyes(self):
        return self._eyes

    def transform(self, vertices):
        if self._T is None:
            vertices = srf.look_at(vertices, self._eye)
        else:
            vertices = srf.world_2_cam(vertices, self._T)
        # perspective transformation
        if self.perspective:
            vertices = perspective(vertices, angle=self.viewing_angle, K = self._K, K_list = self._K_list, img_size=self.img_size_)
        else:
            vertices = orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Look(Transform):
    def __init__(self, camera_direction=[0, 0, 1], perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(Look, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye
        self.camera_direction = camera_direction

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def set_eyes(self, eyes):
        self._eye = eyes

    @property
    def eyes(self):
        return self._eyes

    def transform(self, vertices):
        vertices = srf.look(vertices, self._eye, self.camera_direction)
        # perspective transformation
        if self.perspective:
            vertices = perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = orthogonal(vertices, scale=self.viewing_scale)
        return vertices
