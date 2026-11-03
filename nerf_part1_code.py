import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time

def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input
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
    encodings = []
    for i in range(num_frequencies):
        freq = torch.tensor((2.0 ** i) * np.pi, device=x.device, dtype=x.dtype)
        encodings.append(torch.sin(freq * x))
        encodings.append(torch.cos(freq * x))

# Add the sin/cos part once
    results.append(torch.cat(encodings, dim=-1) if encodings
                  else x.new_empty((x.shape[0], 0)))
    return torch.cat(results, dim=-1)

class model_2d(nn.Module):

    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """

    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        input_dim = (num_frequencies * 2 + 1) * 2  # (2 for x and y)

        self.layer_in = nn.Linear(input_dim, filter_size)
        self.act1 = nn.ReLU()
        self.layer = nn.Linear(filter_size, filter_size)
        self.act2 = nn.ReLU()
        self.layer_out = nn.Linear(filter_size, 3)
        self.act3 = nn.Sigmoid()  # to keep RGB in [0, 1]
							 
								   
						  
		

    def forward(self, x):
        # example of forward through a layer: y = self.layer_in(x)
        x = self.layer_in(x)
        x = self.act1(x)
        x = self.layer(x)
        x = self.act2(x)
        x = self.layer_out(x)
        x = self.act3(x)
        # x = ...

        return x

def normalize_coord(height, width, num_frequencies=6):

    """
    Creates the 2D normalized coordinates, and applies positional encoding to them

    Args:
    height (int): Height of the image
    width (int): Width of the image
    num_frequencies (optional, int): The number of frequencies used in
      the positional encoding (default: 6).

    Returns:
    (torch.Tensor): Returns the 2D normalized coordinates (range in [0, 1]) after applying positional encoding to them. Shape: [height*width, D*(2*num_frequencies+1)].
    """

    x = torch.linspace(0, 1, width)
    y = torch.linspace(0, 1, height)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")  # shape: [H, W]
    coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # shape: [H*W, 2]
			
    embedded_coordinates = positional_encoding(coords, num_frequencies=num_frequencies, incl_input=True)

    return embedded_coordinates

def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):

    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    # Create the 2D normalized coordinates, and apply positional encoding to them
    embedded_coordinates = normalize_coord(height, width, num_frequencies)

    for i in range(iterations+1):
        optimizer.zero_grad()
        # Run one iteration

        # Compute mean-squared error between the predicted and target images. Backprop!
        pred = model2d(embedded_coordinates.to(device))
        pred = pred.reshape(height, width, 3)
        loss = torch.mean((pred - test_img)**2)
        loss.backward()
        optimizer.step()

        # Display images/plots/stats
        if i % display == 0 and show:
            # Calculate psnr
            psnr = 10 * torch.log10(1 / loss)
            # mse_loss = 10 * torch.log10(1 / loss)
            # R = torch.max(test_img)
            # psnr = 10 * torch.log10(R**2 / mse_loss)

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print('Done!')
    torch.save(model2d.state_dict(),'model_2d_' + str(num_frequencies) + 'freq.pt')
    plt.imsave('van_gogh_' + str(num_frequencies) + 'freq.png',pred.detach().cpu().numpy())
    return pred.detach().cpu()