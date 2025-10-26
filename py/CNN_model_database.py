


import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split , Subset
import scipy.io
import math
import matplotlib.pyplot as plt
import torch

import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.exceptions import TrialPruned

import h5py
import numpy as np


def symmetric_matrix_from_params(params):
    """
    Convert 6 parameters to a 4x4 symmetric matrix with 1s on the diagonal.

    Args:
        params: tensor of shape [batch, 6] or [6], order: [a, b, c, d, e, f]

    Returns:
        Tensor of shape [batch, 4, 4] (or [4,4] if single sample)
    """

    # Helper to convert float to complex128
    def to_complex(t):
        return t.to(torch.complex128)

    if params.ndim == 1:
        a, b, c, d, e, f = params
        mat = torch.tensor([
            [1, a, b, c],
            [a, 1, d, e],
            [b, d, 1, f],
            [c, e, f, 1]
        ], dtype=params.dtype, device=params.device)
    else:
        # Batch version
        a, b, c, d, e, f ,g = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5] , params[:, 6]
        batch_size = params.size(0)
        mat = torch.zeros((batch_size, 4, 4), dtype=params.dtype, device=params.device)
        mat[:, 0, 0] = g
        mat[:, 1, 1] = g
        mat[:, 2, 2] = g
        mat[:, 3, 3] = g
        mat[:, 0, 1] = mat[:, 1, 0] = a
        mat[:, 0, 2] = mat[:, 2, 0] = b
        mat[:, 0, 3] = mat[:, 3, 0] = c
        mat[:, 1, 2] = mat[:, 2, 1] = d
        mat[:, 1, 3] = mat[:, 3, 1] = e
        mat[:, 2, 3] = mat[:, 3, 2] = f
    return to_complex(mat)


class MATLABDataset(Dataset):
    def __init__(self, mat_file_path):
        mat = scipy.io.loadmat(mat_file_path)
        R_coupled_real = torch.tensor(mat['dataset_param_coupling']['R_coupled'][0][0].real,
                                      dtype=torch.float64)  # (4, 4, 500)
        R_coupled_imag = torch.tensor(mat['dataset_param_coupling']['R_coupled'][0][0].imag,
                                      dtype=torch.float64)  # (4, 4, 500)
        R_sig_real = torch.tensor(mat['dataset_param_coupling']['R_sig'][0][0].real, dtype=torch.float64)  # (4, 4, 500)
        R_sig_imag = torch.tensor(mat['dataset_param_coupling']['R_sig'][0][0].imag, dtype=torch.float64)  # (4, 4, 500)
        self.inputs = torch.stack((R_coupled_real, R_coupled_imag), dim=2)  # Shape: (4, 4, 2, 500)
        self.targets = torch.stack((R_sig_real, R_sig_imag), dim=2)  # Shape: (4, 4, 2, 500)
        self.inputs = self._normalize_frobenius(self.inputs)
        self.targets = self._normalize_frobenius(self.targets)
        #
        # with h5py.File(mat_file_path, 'r') as f:
        #     # Assuming structure: f['/dataset_param_coupling/R_coupled/real']
        #
        #     R_coupled_dataset = f['/dataset_param_coupling/R_coupled']
        #     R_coupled_np = np.array(R_coupled_dataset)  # Shape: (100, 4, 4), dtype: void with fields
        #     R_sig_dataset = f['/dataset_param_coupling/R_sig']
        #     R_sig_np = np.array(R_sig_dataset)  # Shape: (100, 4, 4), dtype: void with fields
        #     # Extract real and imaginary parts
        #     R_coupled_real = np.array(R_coupled_np['real'])  # Shape: (100, 4, 4)
        #     R_coupled_imag = np.array(R_coupled_np['imag'])  # Shape: (100, 4, 4)
        #     R_sig_real = np.array(R_sig_np['real'])  # Shape: (100, 4, 4)
        #     R_sig_imag = np.array(R_sig_np['imag'])  # Shape: (100, 4, 4)
        #
        #     # Convert to tensors and permute to match your model shape (C, H, W, N) → (4, 4, 2, 100)
        #     R_coupled_real = torch.tensor(R_coupled_real, dtype=torch.float32)
        #     R_coupled_imag = torch.tensor(R_coupled_imag, dtype=torch.float32)
        #     R_sig_real = torch.tensor(R_sig_real, dtype=torch.float32)
        #     R_sig_imag= torch.tensor(R_sig_imag, dtype=torch.float32)



        # Combine real and imaginary parts into a single tensor with shape (4, 4, 2, 500)

        #
        # self.inputs = self.inputs.permute(3, 1, 2, 0)  # From (100, 4, 2, 4) → (4, 4, 2, 100)
        # self.targets = self.targets.permute(3, 1, 2, 0)  # From (100, 4, 2, 4) → (4, 4, 2, 100)

        # Extract real and imaginary parts of R_coupled and R_sig

        # Combine real and imaginary parts into a single tensor with shape (4, 4, 2, 500)
        # self.inputs = torch.stack((R_coupled_real, R_coupled_imag), dim=2)  # Shape: (4, 4, 2, 500)
        # self.targets = torch.stack((R_sig_real, R_sig_imag), dim=2)  # Shape: (4, 4, 2, 500)

        # Apply Frobenius normalization to each matrix (sample-wise)

    def __len__(self):
        return self.inputs.shape[3]  # Number of samples (500)

    def _normalize_frobenius(self, matrices):
        # matrices: shape (4, 4, 2, N)
        real = matrices[..., 0, :]  # (4, 4, N)
        imag = matrices[..., 1, :]  # (4, 4, N)

        # Frobenius norm: sqrt(real^2 + imag^2).sum()
        frob_norm = torch.sqrt(real.pow(2).sum(dim=(0, 1)) + imag.pow(2).sum(dim=(0, 1)))  # Shape: (N,)

        # Avoid division by zero
        #frob_norm = frob_norm.clamp(min=1e-20)

        # Normalize
        real_normalized = real / frob_norm
        imag_normalized = imag / frob_norm

        # Re-stack back to shape (4, 4, 2, N)
        return torch.stack((real_normalized, imag_normalized), dim=2)

    def __getitem__(self, idx):
        # Get the data corresponding to the idx-th sample
        input_sample = self.inputs[..., idx]  # Shape: (4, 4, 2)
        target_sample = self.targets[..., idx]  # Shape: (4, 4, 2)
        # Permute to get shape (2, 4, 4)
        input_sample = input_sample.permute(2, 0, 1)  # Change to (2, 4, 4)
        target_sample = target_sample.permute(2, 0, 1)  # Change to (2, 4, 4)
        return input_sample, target_sample
class ResidualBlockWithoutBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlockWithoutBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,dtype=torch.float64)
        # No BatchNorm here, only Conv2D and ReLU

    def forward(self, x):
        res = x  # Skip connection (no need to clone)
        x = self.conv(x)
        x = torch.relu(x)  # ReLU activation
        x = x + res  # Adding residual connection
        return x
# ConvBlock: Conv2D -> BatchNorm -> ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same',dtype=torch.float64)
        self.bn = nn.BatchNorm2d(out_channels,dtype=torch.float64)  # BatchNorm2d for normalization across batch and channels
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)  # Normalizing across the batch dimension
        x = self.activation(x)
        return x

class ConvBlock_Wbn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock_Wbn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same',dtype=torch.float64)
        self.bn = nn.BatchNorm2d(out_channels,dtype=torch.float64)  # BatchNorm2d for normalization across batch and channels
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # Normalizing across the batch dimension
        x = self.activation(x)
        return x

# ResidualBlock: Conv2D -> BatchNorm -> ReLU + Input
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same',dtype=torch.float64)
        self.bn = nn.BatchNorm2d(out_channels,dtype=torch.float64)  # BatchNorm2d for normalization across batch and channels

    def forward(self, x):
        res = x  # Skip connection (no need to clone)
        x = self.conv(x)
        x = self.bn(x)  # Normalizing across the batch dimension
        x = torch.relu(x)
        x = x + res  # Adding residual connection
        return x

# New Residual Block without BatchNorm


# CNN with residual blocks
class MyRealCNN(nn.Module):
    def __init__(self , conv1_out_channels=4,conv2out_channel=8,kernel_sz=3,num_res_blocks=0,num_fc=0):
        super(MyRealCNN, self).__init__()
        # Initial ConvBlock layers
        self.conv1 = ConvBlock_Wbn(in_channels=2, out_channels=conv1_out_channels, kernel_size=kernel_sz)
        self.conv2 = ConvBlock(in_channels=conv1_out_channels, out_channels=conv2out_channel, kernel_size=kernel_sz)

        # Fully connected layer to transform output into the desired shape
        # Assuming you want to output a matrix of size (2, 4, 4)
        self.fc = nn.Linear(conv2out_channel * 4 * 4, 2 * 4 * 4,dtype=torch.float64)  # Adjusting the output size
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(conv2out_channel,conv2out_channel, kernel_size=kernel_sz) for _ in range(num_res_blocks)]
        )

        # Compute FC input size assuming fixed output dims (8, 4, 4)
        flattened_size = conv2out_channel * 4 * 4

        # Fully connected layers before reshaping
        fc_layers = []
        in_features = flattened_size
        for _ in range(num_fc):
            fc_layers.append(nn.Linear(in_features, in_features,dtype=torch.float64))
            fc_layers.append(nn.ReLU())

        self.fc_layers = nn.Sequential(*fc_layers)

        self.fc_out = nn.Linear(conv2out_channel * 4 * 4, 2 * 4 * 4,dtype=torch.float64)


    def forward(self, x):
        # Convolutional layers
        input_mat=x[:, 0, :] + 1j * x[:, 1, :]
        x = self.conv1(x)
        x = self.conv2(x)

        # Residual blocks (only applied if num_res_blocks > 0)
        if len(self.res_blocks) > 0:
            x = self.res_blocks(x)


        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  # Flatten into a 1D vector

        # Optional intermediate FC layers
        if len(self.fc_layers) > 0:
            x = self.fc_layers(x)


        # Fully connected layer
        x = self.fc(x)  # Output will be a vector of size (batch_size, 32)
        x = x.view(x.size(0), 2, 4, 4)
        output_matrix = x[:, 0, :] + 1j * x[:, 1, :]
        result = torch.matmul( output_matrix,  input_mat)  # Matrix multiplication
        # Reshape back to 4x4x8 before passing to residual blocks

        # Separate the real and imaginary parts
        result_real = result.real  # Shape: (batch_size, 1, 100000)
        result_imag = result.imag  # Shape: (batch_size, 1, 100000)

        # Stack the real and imaginary parts
        result = torch.stack(( result_real, result_imag), dim=1)  # Shape: (batch_size, 2, 100000)
        result = torch.squeeze(result, dim=2)

        Z_inv_real = output_matrix.real  # Shape: (batch_size, 1, 100000)
        Z_inv_imag = output_matrix.imag  # Shape: (batch_size, 1, 100000)

        # Stack the real and imaginary parts
        Z_inv = torch.stack((Z_inv_real, Z_inv_imag), dim=1)  # Shape: (batch_size, 2, 100000)
        Z_inv = torch.squeeze(Z_inv, dim=2)

        return result,Z_inv

class Reduced_param_CNN(nn.Module):
    def __init__(self , conv1_out_channels=4,conv2out_channel=8,kernel_sz=3,num_res_blocks=0,num_fc=0):
        super(Reduced_param_CNN, self).__init__()
        # Initial ConvBlock layers
        self.conv1 = ConvBlock_Wbn(in_channels=2, out_channels=conv1_out_channels, kernel_size=kernel_sz)
        self.conv2 = ConvBlock(in_channels=conv1_out_channels, out_channels=conv2out_channel, kernel_size=kernel_sz)

        # Fully connected layer to transform output into the desired shape
        # Assuming you want to output a matrix of size (2, 4, 4)
        self.fc = nn.Linear(conv2out_channel * 4 * 4, 2 * 4 * 4)  # Adjusting the output size
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(conv2out_channel,conv2out_channel, kernel_size=kernel_sz) for _ in range(num_res_blocks)]
        )

        # Compute FC input size assuming fixed output dims (8, 4, 4)
        flattened_size = conv2out_channel * 4 * 4

        # Fully connected layers before reshaping
        fc_layers = []
        in_features = flattened_size
        for _ in range(num_fc):
            fc_layers.append(nn.Linear(in_features, in_features))
            fc_layers.append(nn.ReLU())

        self.fc_layers = nn.Sequential(*fc_layers)

        self.fc_out = nn.Linear(conv2out_channel * 4 * 4, 7,dtype=torch.float64)


    def forward(self, x):
        # Convolutional layers
        input_mat=x[:, 0, :] + 1j * x[:, 1, :]
        x = self.conv1(x)
        x = self.conv2(x)

        # Residual blocks (only applied if num_res_blocks > 0)
        if len(self.res_blocks) > 0:
            x = self.res_blocks(x)


        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  # Flatten into a 1D vector

        # Optional intermediate FC layers
        if len(self.fc_layers) > 0:
            x = self.fc_layers(x)


        # Fully connected layer
        x = self.fc_out(x)  # Output will be a vector of size (batch_size, 32)

        output_matrix = symmetric_matrix_from_params(x)
        result = torch.matmul( output_matrix,  input_mat)  # Matrix multiplication
        # Reshape back to 4x4x8 before passing to residual blocks

        # Separate the real and imaginary parts
        result_real = result.real  # Shape: (batch_size, 1, 100000)
        result_imag = result.imag  # Shape: (batch_size, 1, 100000)

        # Stack the real and imaginary parts
        result = torch.stack(( result_real, result_imag), dim=1)  # Shape: (batch_size, 2, 100000)
        result = torch.squeeze(result, dim=2)

        Z_inv_real = output_matrix.real  # Shape: (batch_size, 1, 100000)
        Z_inv_imag = output_matrix.imag  # Shape: (batch_size, 1, 100000)

        # Stack the real and imaginary parts
        Z_inv = torch.stack((Z_inv_real, Z_inv_imag), dim=1)  # Shape: (batch_size, 2, 100000)
        Z_inv = torch.squeeze(Z_inv, dim=2)

        return result,Z_inv