import pickle

import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from LSTM_models import SequentialRNNNet
from Matlab_dataloaders import MATLABDataset_inter,MATLABDataset_spectogram
from gen_utils import save_torch_model,load_torch_model
from CNN_spectogram_models import IQ_spectogram_CNN_res_skipconn,test_reconstract
def train_network(mat_file_path,sig_len ,num_epochs, learning_rate=0.01, batch_size=1,\
                   validation_split=0.2,device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f'torch.cuda.is_available():{torch.cuda.is_available()}')
    dataset = MATLABDataset_spectogram(mat_file_path,signal_len=sig_len)
    # with open("dataset_spectogram_iq", 'wb') as f:
    #     pickle.dump(dataset, f)

    dataset.set_train()
    train_dataset = dataset.get_train_set()
    dataset.set_validation()
    val_dataset = dataset.get_validation_set()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model = IQ_spectogram_CNN_res_skipconn()
    # model = load_torch_model(model=model,path='IQ_spectrogram_CNN_epoch_1.pth' )
   # test_reconstract(model=model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0,cooldown=1)
    lr=learning_rate


    total_params = sum(p.numel() for p in model.parameters())  # Total number of parameters
    total_size = total_params * 4  # Each parameter is typically stored as a 32-bit float (4 bytes)
    size_in_MB = total_size / (1024 ** 2)
    print(f"Model size: {size_in_MB:.2f} MB")
    #n_train_batches=10
    #n_train_batches=5
    for epoch in range(num_epochs):
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != lr:
            lr=new_lr
            print("lr is now" + str(lr))

        model.train()
        epoch_loss = 0.0
        cnt=0
        diff=0.0


        for real_clean, imag_clean, real_distor, imag_distor in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            real_distor, imag_distor= real_distor.unsqueeze(1),imag_distor.unsqueeze(1)
            real_clean, imag_clean = real_clean.unsqueeze(1), imag_clean.unsqueeze(1)
            out_real,out_imag = model(real_distor),model(imag_distor)
            loss = criterion(out_real,real_clean)+criterion(out_imag,imag_clean)
            diff += criterion(real_distor,real_clean) + criterion(imag_distor,imag_clean)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss
            # cnt+=1
            # if cnt > n_train_batches:
            #     break

        avg_epoch_loss = epoch_loss /len(train_loader)
        avg_diff = diff / len(train_loader)
        train_metric = avg_epoch_loss/avg_diff
        print(f"Epoch [{epoch + 1}/{num_epochs}], train_metric: {train_metric:}")
        save_torch_model(model=model, path=f'IQ_spectrogram_CNN_epoch_{epoch + 1}.pth')
 #
 # # Validation phase
        model.eval()
        val_loss = 0.0
        diff_val=0.0
        with (torch.no_grad()):
            for real_clean, imag_clean, real_distor, imag_distor in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                real_distor, imag_distor = real_distor.unsqueeze(1), imag_distor.unsqueeze(1)
                real_clean, imag_clean = real_clean.unsqueeze(1), imag_clean.unsqueeze(1)
                out_real, out_imag = model(real_distor), model(imag_distor)
                loss = criterion(out_real, real_clean) + criterion(out_imag, imag_clean)
                diff_val += criterion(real_distor,real_clean) + criterion(imag_distor,imag_clean)
                val_loss += loss
            avg_val_loss = val_loss / len(val_loader)
            avg_diff_val = diff_val / len(val_loader)
            val_metric = avg_val_loss / avg_diff
            print(f"Epoch [{epoch + 1}/{num_epochs}], val_metric: {avg_diff_val:}")
        scheduler.step(val_metric)



 #            #avg_val_loss = val_loss / len(val_loader)
 #            val_ratio = float(val_loss/diff_val)
 #            #print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:}")
 #            print("val ratio" + str(val_ratio))
 #            val_metric = float(val_ratio) #+ float(model_size_mb/10)
 #



train_network(mat_file_path='IQ_400sig_10_02.mat', num_epochs=50, batch_size=5 ,\
              learning_rate=0.0005,sig_len=80000)
# for optuna

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model = load_torch_model(model=model,path='10_02_backprop1_best_siglen1000 ' )
# Print the size of the model (total number of parameters)
# Calculate the total number of parameters (in bytes)
# total_params = model.count_params()
# # for name, param in model.named_parameters():
# #      print(f"{name}: {param.numel()} parameters")
# # Assuming 4 bytes per parameter (float32)
# model_size_bytes = total_params * 4
# model_size_mb = model_size_bytes / (1024 ** 2)
# print(f"Model size: {model_size_mb:.2f} MB")
#
# if model_size_mb > 30 or proj_size>cell_size:
#     raise  optuna.exceptions.TrialPruned()


# if sig_iter >= (n_train_examples/batch_size):
#     sig_iter=0
#     break
#            # Check if the current validation metric is better (lower)
 #            # if val_metric < best_val_ratio:
 #            #     # Save the model if this is the best validation metric so far
 #            #     best_val_ratio = val_metric
 #            #     model_save_path = f"best_model_trial_{trial.number}.pth"
 #            #     torch.save(model.state_dict(), model_save_path)
 #            #     print(f"Saved best model for trial {trial.number}, epoch {epoch + 1}")
 #
 #
 #        if backprop_const > int(sig_len/step_interval):
 #            backprop_const=int(sig_len/step_interval)
 #        trial.report(val_metric ,epoch)
 #        if trial.should_prune():
 #            raise optuna.exceptions.TrialPruned()
 #        scheduler.step(val_metric)
 #            # else:
 #            #     backprop_const+=1
 #            #     print("backward prog is cal every" + str(backprop_const) + 'steps' )
 #
 #        # Increase the step interval for the next epoch
 #    model_save_path = f"final_model_trial_{trial.number}.pth"
 #    torch.save(model.state_dict(), model_save_path)
 #    print(f"Saved last model for trial {trial.number}")
 #    return val_metric




#
# # Create and optimize using Optuna
# with open ("PROJ_RNN_siglen10K_finetune_18_02",'rb') as f:
#     study=pickle.load(f)
# # Convergence plot
#
# fig = optuna.visualization.plot_param_importances(study)
# fig.show()
#
# convergence_fig = optuna.visualization.plot_contour(study, params=["proj_size", "input_sz"])
# convergence_fig.show()
#

