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



import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch.optim as optim
from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class FCO_Estim(nn.Module):
    def __init__(self,input_len=1000,batch_sz=40):
        super(FCO_Estim, self).__init__()
        self.sig_len = input_len
        self.batch_size = batch_sz
        # Encoder layers
        self.FC = nn.Linear(in_features=input_len,out_features=input_len)
        self.FC2 = nn.Linear(in_features=input_len,out_features=input_len)
        self.FCdil_500 = nn.Linear(in_features=input_len,out_features=500)
        self.FCdil_50 = nn.Linear(500, 50)
        self.FC_final = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self._initialize_weights()
        # Decoder layers (upsampling)

    def forward(self, in_sig):


        x=self.FC(in_sig)
        x=self.relu(x)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.FCdil_500(x)
        x = self.relu(x)
        x = self.FCdil_50(x)
        x = self.relu(x)
        x = self.FC_final(x)
        gain=x
        return gain

    def _initialize_weights(self):
        # Initialize all weights and biases to 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # Set all biases to 0

def train_network(mat_file_path, buffer_len,sig_len ,num_epochs, learning_rate=1e-4, batch_size=40,\
                   validation_split=0.2,device='cuda' if torch.cuda.is_available() else 'cpu'):


    dataset = MATLABDataset_inter(mat_file_path, signal_len=sig_len,fco_distor=True)
    dataset.set_train()
    train_dataset = dataset.get_train_set()
    dataset.set_validation()
    val_dataset = dataset.get_validation_set()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FCO_Estim().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1)
    step_interval = 1000  # Initial interval size
    interval_growth =0 # Growth factor for interval size
    grad_flag = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            _,input_sig,fco_tar = batch
            fco_tar = fco_tar.to(device)
            #target_real, target_imag = target_real.to(device), target_imag.to(device)
            utput_signal_real, output_signal_imag = [], []
            loss = 0.0
            sum_fco=0
            # Process the input signal in chunks
            for t in range(0,sig_len,step_interval):
                # Select the chunk for current interval
                input_sig_interval = input_sig[:, t:t + step_interval]
                # Forward pass
                fco_out = model(input_sig_interval)
                sum_fco= sum_fco + fco_out
                # interval_loss = criterion(gain_out,gain_tar)
                # loss = loss + interval_loss
                # # # Clear gradients for the next interval
                # interval_loss.backward()
                # optimizer.step()
                # optimizer.zero_grad()
                # # if t==1000:
                #     print(str(loss) + "\n")
                # Inner scheduler: decrease learning rate within epoch
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] *= 0.99  # Reduce by 0.7
                # Accumulate the interval loss and backpropagate
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] /= 0.99**(sig_len/step_interval)   # increase to original
            gain_pred=sum_fco/(sig_len/step_interval)
            loss = criterion(gain_pred,fco_tar)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)
            epoch_loss += loss
        avg_epoch_loss = epoch_loss /(len(train_loader)*batch_size)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_epoch_loss:}")

        # # Increase the step interval for the next epoch
        # if step_interval<buffer_len:
        #     step_interval = step_interval + interval_growth
        #     print(f"Next step interval: {step_interval} time steps")
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, input_sig, fco_tar = batch
                fco_tar = fco_tar.to(device)
                # Calculate the validation loss in chunks
                loss = 0.0
                sum_gain =0
                for t in range(0, sig_len, step_interval):
                    input_sig_chunk = input_sig[:, t:t + step_interval]

                    # Forward pass
                    gain_out = model(input_sig_chunk)
                    sum_gain = sum_gain + gain_out
                    # Calculate the loss for the current interval
                gain_pred = sum_gain / (sig_len/step_interval)
                val_loss += criterion(gain_pred, fco_tar)

        avg_val_loss = val_loss / (len(val_loader)*batch_size)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Validation Loss: {avg_val_loss:.4f}")



# Example usage within the training function:
# After training is complete, save two signals and their outputs
# Replace 'mat_file_path' and 'signal_len' with appropriate values
#  print("Training complete.")
 #   torch.save(model.state_dict(), 'trained_lstm_net.pth')
#    print("Model saved as 'trained_lstm_net.pth'.")
# Example usage

train_network(mat_file_path='FCO_400_16_02.mat', num_epochs=50, batch_size=40 ,\
              learning_rate=1e-4,buffer_len=1000,sig_len=80000)
# phase 0.72 ~ deg MSE error for 1000 sig len
# phase 0.75 ~ deg MSE error for 10,000 sig len
# gain 0.025 ~ dB MSE error for 1000 sig len
# gain 0.025~ dB MSE error for 10000 sig len
# FCO Khz MSE foor 1000 sig len
# FCO 0.0337 Khz MSE foor 1000 sig len
# FCO 0.0287 Khz MSE foor 10,000 sig len
# FCO 0.032 Khz MSE foor 80,000 full sig len









