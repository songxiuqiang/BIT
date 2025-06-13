import numpy as np
import torch
import torch.nn.functional as F

def look_at(vertices, eye, at=[0, 0, 0], up=[0, 1, 0]):
    print("look at")
    """
    "Look at" transformation of vertices.
    """
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    device = vertices.device

    # if list or tuple convert to numpy array
    if isinstance(at, list) or isinstance(at, tuple):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    # if numpy array convert to tensor
    elif isinstance(at, np.ndarray):
        at = torch.from_numpy(at).to(device)
    elif torch.is_tensor(at):
        at.to(device)

    if isinstance(up, list) or isinstance(up, tuple):
        up = torch.tensor(up, dtype=torch.float32, device=device)
    elif isinstance(up, np.ndarray):
        up = torch.from_numpy(up).to(device)
    elif torch.is_tensor(up):
        up.to(device)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
    elif isinstance(eye, np.ndarray):
        eye = torch.from_numpy(eye).to(device)
    elif torch.is_tensor(eye):
        eye = eye.to(device)

    print('eye.shape with out repeat', eye.shape)	
    # print('eye',eye)
    batch_size = vertices.shape[0]
    if eye.ndimension() == 1:
        eye = eye[None, :].repeat(batch_size, 1)
    if at.ndimension() == 1:
        at = at[None, :].repeat(batch_size, 1)
    if up.ndimension() == 1:
        up = up[None, :].repeat(batch_size, 1)
    print('eye.shape after repeat', eye.shape)	
    

    # create new axes
    # eps is chosen as 1e-5 to match the chainer version
    z_axis = F.normalize(at - eye, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    print('r.shape', r.shape)
    print('r', r)
    	
    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    # print('eye.shape', eye.shape)
    # print('eye', eye)
    # print('vertices',vertices)
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1, 2))

    return vertices

def world_2_cam(vertices, T):
    # print('world_2_cam')
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    device = vertices.device

    if isinstance(T, list) or isinstance(T, tuple):
        T = torch.tensor(T, dtype=torch.float32, device=device)
    elif isinstance(T, np.ndarray):
        T = torch.from_numpy(T).to(device)
    elif torch.is_tensor(T):
        T = T.to(device)

    batch_size = vertices.shape[0]
    
    # print('T.shape with out repeat', T.shape)
    # print('T.ndimension',T.ndimension()) 

    look_at_matrix = np.array([[1, 0, -0, -0],
                               [0, -1, 0, 0],
                               [-0, -0, -1, 0],
                               [0, 0, 0, 1]], dtype = np.float32)
    look_at_matrix = torch.from_numpy(look_at_matrix).to(device)
    if T.ndimension() == 2:
        # T = np.repeat(T[np.newaxis, ...], batch_size, axis=0)
        # T = np.stack(T, axis=0)
        T = T[None, :,:].repeat(batch_size, 1, 1)
    if look_at_matrix.ndimension() == 2:
        look_at_matrix = look_at_matrix[None, :,:].repeat(batch_size, 1, 1)
    # if T.ndimension() == 1:
        # T = T[None, :].repeat(batch_size, 1)
    # print('T.shape after repeat', T.shape)
    # print('T.ndimension',T.ndimension()) 
    
    # T = T.type(torch.FloatTensor)
    # print('look_at_matrix.shape', look_at_matrix.shape)
    # print('T.shape', T.shape)
    # print('type(look_at_matrix)', type(look_at_matrix), look_at_matrix.dtype)
    # print('type(T)', type(T),  T.dtype)
    # print('type(vertices)', type(vertices),  vertices.dtype)
    # print('T', T)
    # print('vertices', vertices)
    T = torch.matmul(look_at_matrix, T)
    R = T[:,:3,:3]
    R = R.float()
    
    t = T[:,:3,3]
    t = t.float() 
    
    # print('R.shape', R.shape, 't.shape', t.shape)
    vertices = torch.matmul(vertices, R.transpose(1, 2))
    if vertices.shape != t.shape:
        t = t[:, None, :]
        
    # print('R.shape', R.shape, 't.shape', t.shape)
    # print('R',R)
    # print('t',t)
    # print('vertices',vertices)
    vertices = vertices + t
    # print('vertices by T',vertices)
    return vertices