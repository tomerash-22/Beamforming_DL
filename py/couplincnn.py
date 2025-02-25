import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io
import math
import matplotlib.pyplot as plt
import torch


def custom_loss_function(output, target, reg_factor_H):
    # Separate real and imaginary parts
    real_part = output[:, 0, :, :]
    imag_part = output[:, 1, :, :]
    # Reconstruct complex matrix A
    A = torch.complex(real_part, imag_part)
    # Compute the conjugate transpose of A
    A_H = A.conj().transpose(-2, -1)
    # Using transpose for last two dims to get (4, 4) transpose
    # Difference matrix A - A_H
    difference = A - A_H
    # Sum over batch of element-wise squared L2 norm for Hermitian regularization
    regularization_term = torch.sum(difference.abs() ** 2, dim=(1, 2)).mean() # Mean over batch
    # Mean squared error loss between output and target
    mse_loss = torch.sum((output - target) ** 2)
    # Combine MSE loss and Hermitian regularization term
    total_loss = mse_loss + reg_factor_H * regularization_term

    return total_loss


class MATLABDataset(Dataset):
    def __init__(self, mat_file_path):
        mat = scipy.io.loadmat(mat_file_path)

        # Extract real and imaginary parts of R_coupled and R_sig
        R_coupled_real = torch.tensor(mat['dataset_param_coupling']['R_coupled'][0][0].real, dtype=torch.float32)  # (4, 4, 500)
        R_coupled_imag = torch.tensor(mat['dataset_param_coupling']['R_coupled'][0][0].imag, dtype=torch.float32)  # (4, 4, 500)
        R_sig_real = torch.tensor(mat['dataset_param_coupling']['R_sig'][0][0].real, dtype=torch.float32)  # (4, 4, 500)
        R_sig_imag = torch.tensor(mat['dataset_param_coupling']['R_sig'][0][0].imag, dtype=torch.float32)  # (4, 4, 500)

        # Combine real and imaginary parts into a single tensor with shape (4, 4, 2, 500)
        self.inputs = torch.stack((R_coupled_real, R_coupled_imag), dim=2)  # Shape: (4, 4, 2, 500)
        self.targets = torch.stack((R_sig_real, R_sig_imag), dim=2)  # Shape: (4, 4, 2, 500)

    def __len__(self):
        return self.inputs.shape[3]  # Number of samples (500)

    def __getitem__(self, idx):
        # Get the data corresponding to the idx-th sample
        input_sample = self.inputs[..., idx]  # Shape: (4, 4, 2)
        target_sample = self.targets[..., idx]  # Shape: (4, 4, 2)
        # Permute to get shape (2, 4, 4)
        input_sample = input_sample.permute(2, 0, 1)  # Change to (2, 4, 4)
        target_sample = target_sample.permute(2, 0, 1)  # Change to (2, 4, 4)
        return input_sample, target_sample

# ConvBlock: Conv2D -> BatchNorm -> ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # BatchNorm2d for normalization across batch and channels
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)  # Normalizing across the batch dimension
        x = self.activation(x)
        return x

class ConvBlock_Wbn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock_Wbn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # BatchNorm2d for normalization across batch and channels
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # BatchNorm2d for normalization across batch and channels

    def forward(self, x):
        res = x  # Skip connection (no need to clone)
        x = self.conv(x)
        x = self.bn(x)  # Normalizing across the batch dimension
        x = torch.relu(x)
        x = x + res  # Adding residual connection
        return x

# New Residual Block without BatchNorm
class ResidualBlockWithoutBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlockWithoutBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # No BatchNorm here, only Conv2D and ReLU

    def forward(self, x):
        res = x  # Skip connection (no need to clone)
        x = self.conv(x)
        x = torch.relu(x)  # ReLU activation
        x = x + res  # Adding residual connection
        return x

# CNN with residual blocks
class MyRealCNN(nn.Module):
    def __init__(self):
        super(MyRealCNN, self).__init__()
        # Initial ConvBlock layers
        self.conv1 = ConvBlock_Wbn(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(in_channels=4, out_channels=8, kernel_size=3, padding=1)

        # Fully connected layer to transform output into the desired shape
        # Assuming you want to output a matrix of size (2, 4, 4)
        self.fc = nn.Linear(8 * 4 * 4, 2 * 4 * 4)  # Adjusting the output size

    def forward(self, x):
        # Convolutional layers
        input_mat=x[:, 0, :] + 1j * x[:, 1, :]
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  # Flatten into a 1D vector
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
        return result


# def l2_loss(output, target):
#     loss = nn.MSELoss()
#     return loss(output, target)

def train_model(model, train_loader, val_loader, optimizer, threshold, max_epochs, batch_sz,gamma,step_size):
    # Initialize lists to store the losses
    training_losses = []
    validation_losses = []
    grad_norm_lst = []

    # Track the best validation loss and model
    best_val_loss = float('inf')  # Start with infinity to ensure any loss will be better
    best_model_state = None  # To store the model state with the best validation loss

    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        num_batches = 0
        total_grad_norm = 0.0  # To accumulate gradient norms
        reg_diag = 1e-9
        # Training
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            # inputs = inputs.unsqueeze(0)  # Commented out
            # targets = targets.unsqueeze(0)  # Commented out
            output = model(inputs)  # Forward pass
            loss = custom_loss_function(output, targets,reg_diag)  # Calculate loss
            loss.backward()  # Backpropagation
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10e4)  # Clip gradients to a max norm  # Commented out

            optimizer.step()  # Update weights
            grad_norm = 0.0
            param_cnt = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    param_cnt += 1
            if param_cnt > 0:
                total_grad_norm += grad_norm / param_cnt  # Average gradient norm

            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        grad_norm_lst.append(total_grad_norm)

        # Validation
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                output = model(inputs)  # Forward pass
                loss = custom_loss_function(output, targets,reg_diag)  # Calculate loss  # Calculate loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{max_epochs}], "
              f"Train Loss: {avg_train_loss:}, "
              f"Validation Loss: {avg_val_loss:}, "
              f"Avg Grad Norm: {total_grad_norm:}")
        # Check if this is the best model (lowest validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save the model state

        # Early stopping condition
        if avg_val_loss < threshold:
            print("Validation loss below threshold, stopping training.")
            for i in range(3):  # Print 3 different matrices
                input_sample, target_sample = dataset[i]  # Get input and target from dataset
                input_sample = input_sample.unsqueeze(0)  # Add batch dimension for model input
                output_sample = model(input_sample)  # Get model output

                # Convert to numpy for better visualization
                input_matrix = input_sample.squeeze().numpy()  # Remove batch dimension
                output_matrix = output_sample.detach().squeeze().numpy()  # Detach and remove batch dimension
                target_matrix = target_sample.numpy()  # Convert target to numpy

                print(f"Input Matrix {i + 1}:\n{input_matrix}")
                print(f"Output Matrix {i + 1}:\n{output_matrix}")
                print(f"Target Matrix {i + 1}:\n{target_matrix}\n")
            break

    return training_losses, validation_losses, grad_norm_lst

# Load dataset and create DataLoader

mat_file_path = 'data_set_remote_desk_ses.mat'  # Path to your .mat file
dataset = MATLABDataset(mat_file_path)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_sz = 10  # You can adjust this as needed
train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

# Initialize model, optimizer, and training parameters
gamma = 0.5
step_size=20
model = MyRealCNN()
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Set your initial learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
threshold = 5e-10  # You can adjust the threshold
max_epochs = 15 # Maximum number of epochs

# Train the model
training_losses, validation_losses, grad_norm_lst = \
train_model(model, train_loader, val_loader, optimizer,
            threshold, max_epochs, batch_sz,gamma,step_size)

# Plot losses
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# If desired, save the model
# torch.save(model.state_dict(), 'my_real_cnn.pth')
