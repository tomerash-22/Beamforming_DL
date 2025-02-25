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
from Matlab_dataloaders import MATLABDataset_inter
from gen_utils import save_torch_model,load_torch_model
import optuna
import optuna.visualization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
epochs = 10
mat_file_path = 'IQ_400sig_10_02.mat'
num_epochs = 30
buffer_len = 1000
sig_len = 80000
backprop_const = 1
n_train_examples = 320
n_valid_examples = 80
num_dir=2

best_val_ratio=2

# Define the objective function for Optuna optimization
def objective(trial):
    # Define the lists of possible values for the hyperparameters
    #sig_2fc_values = range(40,200,2)  # List of possible values for sig_2fc
    in_2sigfc_ratio_vals = [17]
    backprop_sced_val = range(1,5)
    lr_vals = np.logspace(-7,-3.5,100)
    input_size_vals =[30] #nrange(24,48,2)
    proj_size_values = [60]  #range(40,80,10)  # List of possible values for proj_size
    cell_size_values = [2200] # range(1800,3000,100) # List of possible values for cell_size
    batch_size_vals = [2,5,8,10,20,40]
    sced_fact_vals = np.linspace(0.1,0.9,10)
    Wd_vals = np.logspace(-7,-4.5,20)
    #lr_vals =  [0.007]
    # backprop_val = range(10)
    RNN_type_vals = ['LSTM']#, 'RNN', 'GRU']

    proj_size = trial.suggest_categorical("proj_size", proj_size_values)
    cell_size = trial.suggest_categorical("cell_size", cell_size_values)
    in_2sigfc_ratio = trial.suggest_categorical("input2FC_ratio", in_2sigfc_ratio_vals)
    input_size = trial.suggest_categorical("input_sz",  input_size_vals)
    W_decay = trial.suggest_categorical("w_decay",  Wd_vals)
    RNN_type = trial.suggest_categorical("rnn_type", RNN_type_vals)
    learning_rate=trial.suggest_categorical("lr", lr_vals)
    num_dir=trial.suggest_categorical("bi", [1])
    backprop_sced_const=trial.suggest_categorical("backprop_sced",backprop_sced_val)
    batch_size=trial.suggest_categorical("batch_sz",batch_size_vals)
    sced_fact = trial.suggest_categorical("sced_factor",sced_fact_vals)
    sig2fc = input_size * in_2sigfc_ratio
    model = SequentialRNNNet(input_size=input_size,proj_size=proj_size,sig_to_fc=sig2fc,
                             RNNtype=RNN_type,cell_size=cell_size,batch_size=batch_size,bi=num_dir).to(device)
    best_val_ratio=2

# def train_network(mat_file_path, buffer_len,sig_len ,num_epochs, learning_rate=0.01, batch_size=1,\
#                    validation_split=0.2,device='cuda' if torch.cuda.is_available() else 'cpu',
#                   input_size=10,cell_size=1000,num_dir=2,sig_2fc=100,backprop_const=5,proj_size=100,
#                   model_path = '8_02_IQnopass_h_and_c.pth',RNN_type='LSTM',optuna_trail=None):
    backprop_const = 1
    dataset = MATLABDataset_inter(mat_file_path,signal_len=sig_len)
    dataset.set_train()
    train_dataset= dataset.get_train_set()
    dataset.set_validation()
    val_dataset=dataset.get_validation_set()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        # for optuna

    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #model = load_torch_model(model=model,path='10_02_backprop1_best_siglen1000 ' )
    # Print the size of the model (total number of parameters)
    # Calculate the total number of parameters (in bytes)
    total_params = model.count_params()
    # for name, param in model.named_parameters():
    #      print(f"{name}: {param.numel()} parameters")
    # Assuming 4 bytes per parameter (float32)
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 ** 2)
    print(f"Model size: {model_size_mb:.2f} MB")

    if model_size_mb > 30 or proj_size>cell_size:
        raise  optuna.exceptions.TrialPruned()


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=W_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sced_fact, patience=0,cooldown=1)
    step_interval = 2*sig2fc  # Initial interval size
    interval_growth =0  # Growth factor for interval size
    grad_flag = 0
    lr=learning_rate
    best_val = 10e8
    for epoch in range(num_epochs):
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != lr:
            lr=new_lr
            print("lr is now" + str(lr))
            backprop_const += backprop_sced_const

        model.train()
        epoch_loss = 0.0
        epoch_diff=0.0
        sig_iter=0

        #save_torch_model(model=model , path='10_02_backprop1')

        for clean_sig , distor_sig in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            clean_sig , distor_sig = clean_sig.to(device), distor_sig.to(device)
            backprop_iter = 0
            diff=0.0
            sig_loss=0
            if proj_size !=0:
                h_0,c_0 =torch.ones(num_dir,batch_size ,proj_size), torch.ones(num_dir,batch_size ,cell_size)
            else:
                h_0, c_0 = torch.ones(num_dir, batch_size, cell_size), torch.ones(num_dir, batch_size, cell_size)
            # Process the input signal in chunks
            for t in range(0,sig_len,step_interval):
                # Select the chunk for current interval
                input_chunk =  distor_sig[:, t:t + step_interval]
                target_chunk = clean_sig[:, t:t + step_interval]
                backprop_iter += 1
                # Forward pass
                if backprop_iter == backprop_const:
                    if RNN_type=='LSTM':
                        output,h_0,c_0 = model(input_chunk,h_0,c_0 )
                        h_0 , c_0 = h_0.detach(), c_0.detach()
                    else:
                        output, h_0 = model(input_chunk, h_0,c_0)
                        h_0= h_0.detach()
                else:
                    if RNN_type == 'LSTM':
                        output, h_0, c_0 = model(input_chunk, h_0, c_0)
                    else:
                        output, h_0 = model(input_chunk, h_0, c_0)

                diff =  diff+ criterion(input_chunk,target_chunk)
                # Calculate the loss for the current interval
                # target_chunk = target_chunk.flatten()
                loss = criterion(output,target_chunk)
                # Accumulate the interval loss and backpropagate
                sig_loss=sig_loss+loss

                if backprop_iter == backprop_const :
                    epoch_loss += sig_loss
                    sig_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    sig_loss,backprop_iter = 0,0

            #sig_iter+=1
            epoch_diff += diff
            epoch_loss += sig_loss
            # if sig_iter >= (n_train_examples/batch_size):
            #     sig_iter=0
            #     break

        #avg_diff = epoch_diff / len(train_loader)
        #avg_epoch_loss = epoch_loss /len(train_loader)
        train_ratio = float(epoch_loss/epoch_diff)
        #print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_epoch_loss:}")
        print( "train_ratio" + str(train_ratio))


 # Validation phase
        model.eval()
        val_loss = 0.0
        diff_val=0.0
        with (torch.no_grad()):
            for clean_sig, distor_sig in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                clean_sig, distor_sig = clean_sig.to(device), distor_sig.to(device)

                diff = 0.0
                sig_loss = 0
                if proj_size != 0:
                    h_0, c_0 = torch.ones(num_dir, batch_size, proj_size), torch.ones(num_dir, batch_size, cell_size)
                else:
                    h_0, c_0 = torch.ones(num_dir, batch_size, cell_size), torch.ones(num_dir, batch_size, cell_size)

                # Process the input signal in chunks
                for t in range(0, sig_len, step_interval):
                    # Select the chunk for current interval
                    input_chunk = distor_sig[:, t:t + step_interval]
                    target_chunk = clean_sig[:, t:t + step_interval]

                    # Forward pass
                    if RNN_type == 'LSTM':
                        output, h_0, c_0 = model(input_chunk, h_0, c_0)
                    else:
                        output, h_0 = model(input_chunk, h_0, c_0)

                    diff = diff + criterion(input_chunk, target_chunk)
                    # Calculate the loss for the current interval
                    #target_chunk = target_chunk.flatten()
                    loss = criterion(output, target_chunk)
                    # Accumulate the interval loss and backpropagate
                    sig_loss = sig_loss + loss
                diff_val += diff
                val_loss += sig_loss
                #sig_iter += 1
                # if  sig_iter >= ( n_valid_examples/batch_size):
                #     sig_iter = 0
                #     break
            #avg_val_loss = val_loss / len(val_loader)
            val_ratio = float(val_loss/diff_val)
            #print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:}")
            print("val ratio" + str(val_ratio))
            val_metric = float(val_ratio) #+ float(model_size_mb/10)

            # Check if the current validation metric is better (lower)
            # if val_metric < best_val_ratio:
            #     # Save the model if this is the best validation metric so far
            #     best_val_ratio = val_metric
            #     model_save_path = f"best_model_trial_{trial.number}.pth"
            #     torch.save(model.state_dict(), model_save_path)
            #     print(f"Saved best model for trial {trial.number}, epoch {epoch + 1}")


        if backprop_const > int(sig_len/step_interval):
            backprop_const=int(sig_len/step_interval)
        trial.report(val_metric ,epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        scheduler.step(val_metric)
            # else:
            #     backprop_const+=1
            #     print("backward prog is cal every" + str(backprop_const) + 'steps' )

        # Increase the step interval for the next epoch
    model_save_path = f"final_model_trial_{trial.number}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved last model for trial {trial.number}")
    return val_metric




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



timeout= 60*60*24
study = optuna.create_study(direction='minimize')  # You can change the direction depending on the objective
#study.enqueue_trial({"backprop_sced":1,"batch_sz":20 ,"sced_factor":0.5 })
study.optimize(objective, n_trials=1000 ,timeout=timeout)  # Number of trials to perform

with open ("FULL_run_20_02",'wb') as f:
    pickle.dump(study,f)



pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print("Study statistics: ")
print(" Number of finished trials: ", len(study.trials))
print(" Number of pruned trials: ", len(pruned_trials))
print(" Number of complete trials: ", len(complete_trials))
print("Best trial:")
trial = study.best_trial
print(" Value: ", trial.value)
print(" Params: ")
for key, value in trial.params.items():
 print(" {}: {}".format(key, value))

fig = optuna.visualization.plot_param_importances(study)
fig.show()

# train_network(mat_file_path='IQ_400sig_10_02.mat', num_epochs=50, batch_size=1 ,\
#               learning_rate=0.0005,buffer_len=1000,sig_len=10000,proj_size=50,cell_size=2000
#               ,backprop_const=3,input_size=10)
