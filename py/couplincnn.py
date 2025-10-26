
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optuna import trial
from torch.utils.data import Dataset, DataLoader, random_split , Subset
import scipy.io
import math
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.exceptions import TrialPruned
from CNN_model_database import  MATLABDataset ,MyRealCNN , Reduced_param_CNN
from optuna.visualization import plot_param_importances
#import matplotlib.pyplot as plt  # Optional, for inline display in some setups
from optuna.visualization import plot_contour
import random
import os
def estimate_model_size_mb(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    for buffer in model.buffers():
        total_params += buffer.numel()
    size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
    return size_mb



def objective(trial):
    # --- Hyperparameters to optimize ---

    batch_sz_val = range(5, 40, 5)
    conv2_out_vals = range(6,100,2)
    resblock_val = range(10)
    fc_num_val = range(5)

    conv2_out = trial.suggest_categorical("conv2_out",  conv2_out_vals)
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    lr = 10e-4 #trial.suggest_loguniform("lr", 1e-5, 1e-3)
    gamma = 0.5#tria    l.suggest_categorical("gamma", [0.1, 0.5, 0.7])
      # You can tune this too if needed
    batch_sz = 100  #trial.suggest_categorical("batch_size",batch_sz_val)
    fc_num= trial.suggest_categorical("fc_num",fc_num_val)
    res_num = trial.suggest_categorical("resblock_num", resblock_val)

    # --- Create model ---
    model = MyRealCNN(conv1_out_channels = int(conv2_out/2), conv2out_channel=conv2_out,kernel_sz=kernel_size
                      ,num_res_blocks= res_num,num_fc=fc_num)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Loaders ---
    mat_file_path = 'final_coupling_training_dataset_06_09.mat'  # Path to your .mat file
    # test_model_on_mat(model_path='coupling_cnn.pth' , mat_input_path=mat_file_path,mat_output_path='results_Rsig')

    dataset = MATLABDataset(mat_file_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Define fixed subsets: first part for training, last part for validation
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

    # --- Train ---
    avg_val_metric = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        threshold=10e-400,         # Early stopping threshold
        max_epochs=20,          # Shorter for faster trials
        batch_sz=batch_sz,
        gamma=gamma,
        step_size=1,
        reg_diag=0,
        kappa_reg=10e-10,
        trial=trial# Not used with ReduceLROnPlateau
    )

    # --- Return best validation loss as objective ---
    return avg_val_metric

def test_model_on_mat(model_path, mat_input_path, mat_output_path, input_key='R_coupled', output_key='predicted_R_sig'):
    """
    Load a trained PyTorch model and test it on data from a .mat file, saving predictions to a new .mat file.

    Args:
        model_path (str): Path to the saved PyTorch model (.pt or .pth).
        mat_input_path (str): Path to the input .mat file containing the test data.
        mat_output_path (str): Path where the output .mat file will be saved.
        input_key (str): Key for the input matrix inside the .mat file (default is 'R_coupled').
        output_key (str): Key to use for the output matrix when saving to .mat file (default is 'predicted_R_sig').
    """
    # # Enqueue a specific trial

    ## # Enqueue a specific trial
# study.enqueue_trial({
#     "conv2_out": 38,
#     "kernel_size": 3,
#     "fc_num": 1,
#     "resblock_num": 3
# })

    state_dict = torch.load(model_path)
    for key, value in state_dict.items():
         print(f"{key} → shape: {value.shape}")

    model = MyRealCNN(conv1_out_channels=int(40/2) ,conv2out_channel=40 , kernel_sz=3 ,
                  num_res_blocks=1 ,num_fc = 0)
    # --- Create model ---
    #model = MyRealCNN(conv1_out_channels=int(10/2) ,conv2out_channel=10 , kernel_sz=3 ,
    #              num_res_blocks=0 ,num_fc = 0)
    model = model.double()
    #model = MyRealCNN(conv1_out_channels=int(20 / 2), conv2out_channel=20, kernel_sz=3, num_res_blocks=0, num_fc=4)
    #model = MyRealCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    model.eval()
    # Load dataset
    dataset = MATLABDataset(mat_input_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Run inference
    predictions = []
    Z_inv_all = []

    with torch.no_grad():
        for input_batch, _ in dataloader:
            output_batch, Z_inv_batch = model(input_batch)
            predictions.append(output_batch)
            Z_inv_all.append(Z_inv_batch)

    # Stack predictions and convert to numpy
    predictions = torch.cat(predictions, dim=0)  # Shape: (N, 2, 4, 4)
    Z_inv_all = torch.cat(Z_inv_all, dim=0)  # Shape: (N, 2, 4, 4) or another shape depending on model

    # Convert predictions to complex
    pred_np = predictions.numpy()
    pred_real = pred_np[:, 0, :, :]
    pred_imag = pred_np[:, 1, :, :]
    pred_complex = pred_real + 1j * pred_imag  # Shape: (N, 4, 4)

    # Convert Z_inv to complex (if same format)
    Z_inv_np = Z_inv_all.numpy()
    Z_inv_real = Z_inv_np[:, 0, :, :]
    Z_inv_imag = Z_inv_np[:, 1, :, :]
    Z_inv_complex = Z_inv_real + 1j * Z_inv_imag  # Shape: (N, 4, 4)

    # Transpose to match MATLAB format: (4, 4, N)
    pred_complex_matlab = pred_complex.transpose(1, 2, 0)
    Z_inv_complex_matlab = Z_inv_complex.transpose(1, 2, 0)

    # Save both to .mat file
    scipy.io.savemat(mat_output_path + '.mat', {
        'R_sig_pred': pred_complex_matlab,
        'Z_inv_pred': Z_inv_complex_matlab
    })


# The loss function is scale invarient
# as h hermitian regulaztion factor
def condition_number(matrix):
    # matrix: (batch, M, N) complex tensor
    # Compute singular values
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    # κ = sigma_max / sigma_min
    kappa = S.max(dim=-1).values / S.min(dim=-1).values
    return kappa


def custom_loss_function(output, target, reg_factor_H,reg_kappa = 0.1,
                         target_kappa=1000):
    # Separate real and imaginary parts
    # Separate real and imaginary parts
    real_part = output[:, 0, :, :]
    imag_part = output[:, 1, :, :]
    A = torch.complex(real_part, imag_part)

    real_t = target[:, 0, :, :]
    imag_t = target[:, 1, :, :]
    B = torch.complex(real_t, imag_t)

    # Normalize A and B per sample
    A_flat = A.view(A.shape[0], -1)
    B_flat = B.view(B.shape[0], -1)

    # Find optimal complex scale factor c per sample using least squares:
    #   minimize ||A - c*B||^2 → c = <A, B> / <B, B>
    numerator = torch.sum(B_flat.conj() * A_flat, dim=1)
    #numerator = torch.sum(A_flat.conj() * B_flat, dim=1)
    denominator = torch.sum(B_flat.conj() * B_flat, dim=1)# + 1e-8
    c = numerator / denominator  # shape: (batch,)

    # Expand c to match matrix shape and apply to target
    c = c.view(-1, 1, 1)
    B_scaled = B * c

    l2mse = torch.mean(torch.abs(A - B) ** 2)
    # Scale-invariant loss
    mse = torch.mean(torch.abs(A - B_scaled) ** 2)

    # Hermitian regularization
    A_H = A.conj().transpose(-2, -1)
    diff = A - A_H
    reg_term = torch.mean(torch.abs(diff) ** 2)

    # --- Condition number penalty ---
    kappa = condition_number(A)
    kappa_term = torch.mean(torch.log(kappa))  # log for stability
    #kappa_term = torch.relu(kappa_term - torch.log(torch.tensor(target_kappa)))

    return l2mse #+ reg_kappa*kappa_term #mse +reg_factor_H * reg_term


# def l2_loss(output, target):
#     loss = nn.MSELoss()
#     return loss(output, target)

def train_model(model, train_loader, val_loader, optimizer, threshold, max_epochs,
                batch_sz,gamma,step_size,reg_diag,kappa_reg):#,trial):
    # Initialize lists to store the losses
    training_losses = []
    validation_losses = []
    grad_norm_lst = []
    kappa_vals = []
    # Track the best validation loss and model
    best_val_loss = float('inf')  # Start with infinity to ensure any loss will be better
    best_model_state = None  # To store the model state with the best validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=3,cooldown=3)
    lr=optimizer.param_groups[0]['lr']

    model_size_mb = estimate_model_size_mb(model)
    print(f"Estimated model size: {model_size_mb:.2f} MB")

    if model_size_mb > 10.0:
         print("Model size exceeded 10 MB. Terminating trial.")
         raise optuna.exceptions.TrialPruned()

    for epoch in range(max_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        num_batches = 0
        total_grad_norm = 0.0  # To accumulate gradient norms

        # Training
        diff_train=0.0
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != lr:
            lr = new_lr
            new_lr = optimizer.param_groups[0]['lr']
            print("lr is" + str(new_lr))

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            # inputs = inputs.unsqueeze(0)  # Commented out
            # targets = targets.unsqueeze(0)  # Commented out
            output,_ = model(inputs)  # Forward pass
            loss = custom_loss_function(output, targets,reg_diag,reg_kappa=kappa_reg)  # Calculate loss
            total_loss += loss
            diff_train += custom_loss_function(inputs,targets ,reg_diag,reg_kappa=kappa_reg)
            loss.backward()  # Backpropagation
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10e4)  # Clip gradients to a max norm  # Commented out
            real_part = output[:, 0, :, :]
            imag_part = output[:, 1, :, :]
            A = torch.complex(real_part, imag_part)
            kappa_vals.append(condition_number(A))
            optimizer.step()  # Update weights
            grad_norm = 0.0
            param_cnt = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    param_cnt += 1
            if param_cnt > 0:
                total_grad_norm += grad_norm / param_cnt  # Average gradient norm
            num_batches += 1

        # Calculate average loss for the epoch
        train_metric = total_loss/diff_train
        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        grad_norm_lst.append(total_grad_norm)

        # Validation
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        diff_val = 0.0
        total_val_mse = 0.0
        total_val_kappa = 0.0
        val_batches=0

        with torch.no_grad():
            for inputs, targets in val_loader:
                output,_ = model(inputs)  # Forward pass
                loss = custom_loss_function(output, targets,reg_diag,reg_kappa=kappa_reg)  # Calculate loss  # Calculate loss
                diff_val += custom_loss_function(inputs, targets,reg_diag,reg_kappa=kappa_reg)
                total_val_loss += loss
                # scale-invariant MSE
                real_part = output[:, 0, :, :]
                imag_part = output[:, 1, :, :]
                A = torch.complex(real_part, imag_part)

                real_t = targets[:, 0, :, :]
                imag_t = targets[:, 1, :, :]
                B = torch.complex(real_t, imag_t)

                A_flat = A.view(A.shape[0], -1)
                B_flat = B.view(B.shape[0], -1)
                c = torch.sum(B_flat.conj() * A_flat, dim=1) / torch.sum(B_flat.conj() * B_flat, dim=1)
                c = c.view(-1, 1, 1)
                B_scaled = B * c
                mse_si = torch.mean(torch.abs(A - B_scaled) ** 2)
                # condition number
                kappa = condition_number(A).mean()
                total_val_mse += mse_si.item()
                total_val_kappa += kappa.item()
                val_batches += 1


            avg_val_metric = total_val_loss/ diff_val
            avg_val_loss = total_val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            avg_val_mse = total_val_mse / val_batches
            avg_val_kappa = total_val_kappa / val_batches
            scheduler.step( avg_val_metric)
            print(f"Epoch [{epoch + 1}/{max_epochs}], "
                  f"Avg train metric: {train_metric} , "
                  f"Avg val metric: {avg_val_metric}, "
                  )
            print(
                  f"Val SI-MSE={avg_val_mse:.4e}, "
                    f"Val κ={avg_val_kappa:.2f}"
                  )
            # Check if this is the best model (lowest validation loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()  # Save the model state
                torch.save(model.state_dict(), 'l2_mse_nofrob.pth')
                print ('model save')
            #trial.report(avg_val_metric,epoch)
            #if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

            # Early stopping condition
            if avg_val_metric < threshold:
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

    return avg_val_metric

# Load dataset and create DataLoader

#study = optuna.create_study(direction='minimize')
#T_optim = 6*60*60
# # Enqueue a specific trial
# study.enqueue_trial({
#     "conv2_out": 38,
#     "kernel_size": 3,
#     "fc_num": 1,
#     "resblock_num": 3
# })
#
#study.optimize(objective, n_trials=1000,timeout=T_optim)
#
#
#with open('optuna_study_zero_kappa.pkl', 'wb') as f:
#     pickle.dump(study, f)
#
# print("Study saved to optuna_study_04_06_coupling.pkl.pkl")
#
# with open('optuna_study_04_06_coupling.pkl', 'rb') as f:
#     study = pickle.load(f)
#
# print("Best trial:")
# best_trial = study.best_trial
#
# print(f"  Value (objective): {best_trial.value}")
# print("  Params:")
# for key, value in best_trial.params.items():
#     print(f"    {key}: {value}")
#
#
# fig = plot_contour(study, params=["conv2_out", "resblock_num"])
# fig.show()

# # # #
# # # %%


# Paths
workspace_dir = "pth_final"        # 🔹 your models directory
results_dir = os.path.join(workspace_dir, "results_logonly")
mat_file_path = "coupling_test_16_09.mat"

# Make results directory if not exists
os.makedirs(results_dir, exist_ok=True)

# Loop through all .pth models in workspace
for fname in os.listdir(workspace_dir):
    if fname.endswith(".pth"):
        model_path = os.path.join(workspace_dir, fname)

        # Use model filename (without .pth) as output name
        model_name = os.path.splitext(fname)[0]
        mat_output_path = os.path.join(results_dir, model_name + "_result.mat")

        print(f"Running {model_name} ...")
        test_model_on_mat(model_path=model_path,
                          mat_input_path=mat_file_path,
                          mat_output_path=mat_output_path)



# # # # # # # # # # test_model_on_mat(model_path='coupling_cnn_27_06_reg0_frob_1718.pth' , mat_input_path=mat_file_path,
# # # # # # # # # #                    mat_output_path= '11_07Rsig')

test_model_on_mat(model_path= 'l2_mse_nofrob.pth', mat_input_path=mat_file_path,
                           mat_output_path= 'R_sig_15_09_l2mse_nofrob_1000kappa_val')
# #
# #
#
# # # # used to be
# #mat_file_path='data_set_remote_desk_ses.mat'
# #
#mat_file_path='coupling_16_07_10K.mat'

mat_file_path = 'final_coupling_training_dataset_06_09.mat'
dataset = MATLABDataset(mat_file_path)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Generate all indices and shuffle them
all_indices = list(range(len(dataset)))
#random.shuffle(all_indices)

# Split into train and validation
train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:]
#
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
#
batch_sz = 100  #You can adjust this as needed
train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)
#
# # Initialize model, optimizer, and training parameters
gamma = 0.5
# step_size=10
model = MyRealCNN(conv1_out_channels=int(38/2) ,conv2out_channel=38 , kernel_sz=3 ,
                  num_res_blocks=1 ,num_fc = 1)
model = model.double()
optimizer = optim.Adam(model.parameters(), lr= 10e-4)  # Set your initial learning rate
#model.load_state_dict(torch.load('optuna_sc_zero_kappa.pth'))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
threshold = 10e-400  # You can adjust the threshold
max_epochs = 1000 # Maximum number of epochs
#
# # Train the model
training_losses, validation_losses, grad_norm_lst = \
train_model(model, train_loader, val_loader, optimizer,
             threshold, max_epochs, batch_sz,gamma,step_size=1,reg_diag=0,kappa_reg=10e-11)

# # Plot losses
# plt.plot(training_losses, label='Training Loss')
# plt.plot(validation_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss over Epochs')
# plt.legend()
# plt.show()
#
# # If desired, save the model
# torch.save(model.state_dict(), 'coupling_cnn.pth')
