import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import os
import pickle
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input. 
    '

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor. shape: [N, D*(2*num_frequencies+1)] or [N, D*2*num_frequencies] if incl_input is False.
    """

    results = []
    if incl_input:
        results.append(x)
    # encode input tensor and append the encoded tensor to the list of results.
    for pow in range(num_frequencies):
      results.append(torch.sin(2**pow*np.pi*x))
      results.append(torch.cos(2**pow* np.pi *x))

    return torch.cat(results, dim=-1)

def get_rays(height, width, intrinsics, w_R_c, w_T_c):
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that despite that all rays share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  
    ray_origins = torch.zeros((height, width, 3), device=device) 

    ray_origins[:,:] = w_T_c

    u, v = torch.meshgrid(torch.arange(width, dtype=torch.float32), torch.arange(height, dtype=torch.float32))
    pixels = torch.stack((u.flatten(), v.flatten(), torch.ones_like(u.flatten())))

    Kinv = torch.linalg.inv(intrinsics)

    ray_directions = ((w_R_c @ Kinv @ pixels).T.reshape(height, width, 3)).transpose(0,1)

    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    height, width, _ = ray_origins.shape
    
    depth_points = torch.zeros((height, width, samples))
    ray_points = torch.zeros((height, width, samples, 3))

    

    t = torch.linspace(near, far, samples)

    for i in range(samples):
        ray_points[..., i, :] = ray_origins + t[i] * ray_directions
        depth_points[..., i] = t[i]

    return ray_points, depth_points

class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        num_x_inputs = 3*(1+2*num_x_frequencies)
        num_d_inputs = 3*(1+2*num_d_frequencies)
        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear((num_x_frequencies*2+1)*3, filter_size),
            'layer_2': nn.Linear(filter_size, filter_size),
            'layer_3': nn.Linear(filter_size, filter_size),
            'layer_4': nn.Linear(filter_size, filter_size),
            'layer_5': nn.Linear(filter_size, filter_size),
            'layer_6': nn.Linear(filter_size + num_x_inputs, filter_size),
            'layer_7': nn.Linear(filter_size, filter_size),
            'layer_8': nn.Linear(filter_size, filter_size),
            'layer_s': nn.Linear(filter_size, 1),
            'layer_9': nn.Linear(filter_size, filter_size),
            'layer_10': nn.Linear(filter_size + num_d_inputs, filter_size//2),
            'layer_11': nn.Linear(filter_size//2, 3),
            # the last layer should be like 'layer_xx': nn.Linear(xxx, 3),
        })
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, d):
        # example of forward through a layer: y = self.layers['layer_1'](x)
        # x_input = x
        # x = self.layers['layer_1'](x)
        # ...
        # sigma = ...
        # ...
        # rgb = ...

        y = self.relu(self.layers['layer_1'](x))
        y = self.relu(self.layers['layer_2'](y))
        y = self.relu(self.layers['layer_3'](y))
        y = self.relu(self.layers['layer_4'](y))
        y = self.relu(self.layers['layer_5'](y))

        yx = torch.cat((y,x), dim=-1)
        y = self.relu(self.layers['layer_6'](yx))
        y = self.relu(self.layers['layer_7'](y))
        y = self.relu(self.layers['layer_8'](y))
        sigma = self.layers['layer_s'](y)
        y = self.layers['layer_9'](y)

        yd = torch.cat((y,d), dim=-1)
        y = self.relu(self.layers['layer_10'](yd))
        rgb = self.sigmoid(self.layers['layer_11'](y))
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):

    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    Return Shape:   ray_points_batches: List of chunks of ray points. Each chunk has shape (H, W, S, 3*(2*num_x_frequencies+1)).
                    ray_directions_batches: List of chunks of ray directions. Each chunk has shape (H, W, S, 3*(2*num_x_frequencies+1)).
    """
    # Apply positional encoding to the ray points and directions (you may nomalize the directions here)
    # repeat the directions to match the dimension and number of points S
    H, W, S, _ = ray_points.shape
    norm_rays = torch.linalg.norm(ray_directions, dim=-1)

    ray_directions = ray_directions / norm_rays.unsqueeze(-1)
    populated_directions = ray_directions.unsqueeze(2).expand(-1,-1,S,-1)
    
    ray_points = positional_encoding(ray_points, num_frequencies=num_x_frequencies).reshape(-1, 3*(1+2*num_x_frequencies))
    ray_directions = positional_encoding(populated_directions, num_frequencies=num_d_frequencies).reshape(-1, 3*(1+2*num_d_frequencies))
    ray_directions_batches = get_chunks(ray_directions)
    ray_points_batches = get_chunks(ray_points)
    

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    s: Volume density sigma at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """

    # H, W, N = s.shape

    height, width, samples = s.shape
    print(rgb.shape, s.shape, depth_points.shape, height, width, samples)
    s = torch.relu(s)

    rgb = rgb.reshape(height*width, samples, 3)
    s = s.reshape(height*width, samples)
    depth_points = depth_points.reshape(height*width, samples)

    # deltas = torch.cat((depth_points[:,1:], torch.tensor([10e9]).repeat(height*width, 1)), dim=1) - depth_points
    deltas = torch.cat((depth_points[:,1:], torch.tensor([10e9], device=depth_points.device).repeat(height*width, 1)), dim=1) - depth_points

    # T = torch.exp(-1 * torch.cat([torch.zeros((height*width,1)), torch.cumsum(s * deltas, dim=1)[:,:-1]], dim=1))
    T = torch.exp(-1 * torch.cat([torch.zeros((height*width,1), device=s.device), torch.cumsum(s * deltas, dim=1)[:,:-1]], dim=1))
    bigC = torch.sum((1 - (torch.exp(-1*s * deltas))).unsqueeze(2) * T.unsqueeze(2) * rgb, dim=1)

    # reshape into right shape
    rec_image = bigC.reshape((height, width, 3))

    
    # # Compute the weights of each sample
    # s = nn.functional.relu(s)
    # delta = torch.zeros((H, W, N), device=device)
    # delta[:, :, 0:-1] = depth_points[:, :, 1:] - depth_points[:, :, :-1]#...
    # delta[:, :, -1] = 1e9 
    
    # alpha = 1.0 - torch.exp(-s * delta)#...
    # print(rgb.shape, s.shape, depth_points.shape, alpha.shape)
    # T = torch.ones((height, width, 1), device=device)
    # T = torch.cat([T, torch.cumprod(1.0 - alpha + 1e-10, dim=-1)[..., :-1]], dim=-1)
    
    # #T = ... # hint: the first element of T is 1, the rest are the cumulative product of (1 - alpha) from the last element to the first

    # weights = alpha * T #...
    # # Compute the final color of each pixel
    # # rgb = rgb.reshape(height*width, samples, 3)#...
    
    # # # Normalize the color
    # # rec_image = rgb 
    # # rec_image = rec_image.clamp(0, 1)
    # rec_image = torch.sum(weights.unsqueeze(-1) * rgb, dim=2)  # (height, width, 3)

    # # Normalize and clamp
    # rec_image = rec_image.clamp(0, 1)

    # # height, width, samples = s.shape[:3]
    # # s = torch.relu(s)
    # # rgb = rgb.reshape(height*width, samples, 3)
    # # s = s.reshape(height*width, samples)
    # # depth_points = depth_points.reshape(height*width, samples)
    # # deltas = torch.cat((depth_points[:,1:], torch.tensor([10e9]).repeat(height*width, 1)), dim=1) - depth_points
    # # T = torch.exp(-1 * torch.cat([torch.zeros((height*width,1)), torch.cumsum(s * deltas, dim=1)[:,:-1]], dim=1))
    # # bigC = torch.sum((1 - (torch.exp(-1*s * deltas))).unsqueeze(2) * T.unsqueeze(2) * rgb, dim=1)

    # # rec_image = bigC.reshape((height, width, 3))
    

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):

    # w_R_c = torch.Tensor(pose[:3, :3])
    # w_T_c = torch.Tensor(pose[:3, 3])
    # ray_origins, ray_directions = get_rays(height, width, intrinsics, w_R_c, w_T_c)
    # ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    # ray_points_batches, ray_directions_batches =  get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)

    # rgb = None
    # sigma = None
    # for i in range(len(ray_points_batches)):
    #     rgb_batch, sigma_batch = model(ray_points_batches[i], ray_directions_batches[i])
    #     if rgb is None:
    #         rgb = rgb_batch
    #         sigma = sigma_batch
    #     else:
    #         rgb = torch.cat((rgb, rgb_batch), 0)
    #         sigma = torch.cat((sigma, sigma_batch), 0)
    # # rgb = rgb.reshape(height, width, samples, 3)
    # # sigma = sigma.reshape(height, width, samples)
    # rec_image = volumetric_rendering(rgb, s, depth_points)
    w_R_c = torch.Tensor(pose[:3, :3])
    w_T_c = torch.Tensor(pose[:3, 3])
    ray_origins, ray_directions = get_rays(height, width, intrinsics, w_R_c, w_T_c)
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    ray_points_batches, ray_directions_batches =  get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies)

    # rgb = None
    # sigma = None
    # for i in range(len(ray_points_batches)):
    #     # Ensure batches are on the correct device
    #     rgb_batch, sigma_batch = model(ray_points_batches[i].to(device), ray_directions_batches[i].to(device))
    #     if rgb is None:
    #         rgb = rgb_batch
    #         sigma = sigma_batch
    #     else:
    #         rgb = torch.cat((rgb, rgb_batch), 0)
    #         sigma = torch.cat((sigma, sigma_batch), 0)
    rgb_outputs = []
    s_outputs = []
    for i in range(len(ray_points_batches)):
        # outputs = model(ray_points_batches[i], ray_directions_batches[i])
        outputs = model(ray_points_batches[i], ray_directions_batches[i])
        rgb, s = outputs

        # append to lists for concatenation later on
        rgb_outputs.append(rgb)
        s_outputs.append(s)

    # concat
    rgb = torch.cat(rgb_outputs, dim=0)
    s = torch.cat(s_outputs, dim=0)
          
    rgb = rgb.reshape(height, width, samples, -1)
    s = s.reshape(height, width, samples)
    
    # Apply volumetric rendering to obtain the reconstructed image
    # rec_image = volumetric_rendering(rgb, s, depth_points)
    rec_image = volumetric_rendering(rgb, s, depth_points)
    print(rgb.shape, s.shape, depth_points.shape) 
    # Pass height, width, and samples to volumetric_rendering
    rec_image = volumetric_rendering(rgb, s, depth_points)

    return rec_image