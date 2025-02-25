
import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class SequentialRNNNet(nn.Module):
    def __init__(self, input_size=10, cell_size=1000,num_layers=1,
                 output_size=2000,sig_to_fc=100,proj_size=100
                 ,batch_size=10,RNNtype='LSTM',bi = 2):
        super(SequentialRNNNet, self).__init__()
        if bi==2:
            self.bidir = True
        else:
            self.bidir = False
        self.batch_sz=batch_size
        self.rnn_type=RNNtype
        self.proj_sz = proj_size
        self.out_size = output_size
        self.sig_2fc = sig_to_fc  #must be divisable by output size , th sig len before passing to FC layer
        self.cell_size = cell_size
        self.input_size=input_size

        self.seqL = int (2*self.sig_2fc/self.input_size)
        self.Ext_it = int(output_size/self.sig_2fc)


        if RNNtype=='LSTM':
            if proj_size==0:
                self.RNN = nn.LSTM(input_size=input_size,hidden_size=cell_size,
                                        bidirectional=self.bidir,num_layers=num_layers,batch_first=True)
            else:
                self.RNN = nn.LSTM(input_size=input_size, hidden_size=cell_size,
                                   bidirectional=self.bidir, num_layers=num_layers, batch_first=True
                                   , proj_size=proj_size
                                   )
        elif RNNtype=='GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=cell_size,
                               bidirectional=self.bidir, num_layers=num_layers, batch_first=True)
        elif RNNtype=='RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=cell_size,
                               bidirectional=self.bidir, num_layers=num_layers, batch_first=True)

        if proj_size!=0:
            self.fc_1 = nn.Linear(in_features=self.seqL*self.proj_sz* bi , out_features=self.seqL*self.proj_sz* bi )
            self.bn_1 = nn.BatchNorm1d(num_features=self.seqL *self.proj_sz* bi)
            self.fc_out = nn.Linear(in_features=self.seqL * self.proj_sz * bi, out_features=2 * self.sig_2fc)
            self.bn_2 = nn.BatchNorm1d(num_features=self.seqL * self.proj_sz * bi)

        else:
            self.fc_1 = nn.Linear(in_features=self.seqL * self.cell_size * bi,
                                  out_features=self.seqL * self.cell_size * bi)
            self.bn_1 = nn.BatchNorm1d(num_features=self.seqL *  self.cell_size * bi)
            self.fc_out = nn.Linear(in_features=self.seqL * self.cell_size * bi, out_features=2 * self.sig_2fc)
            self.bn_2 = nn.BatchNorm1d(num_features=self.seqL * self.cell_size * bi)


        #self.drop = nn.Dropout(p=0.2)
        #self.drop_final = nn.Dropout()
        self.relu = nn.ReLU()

        self._initialize_weights()

    def count_params(self):
        total_params = 0
        # Count parameters for the RNN layer
        for name, param in self.RNN.named_parameters():
            if 'weight' in name or 'bias' in name:
                    total_params += param.numel()

        # Count parameters for fully connected layers (fc_1, fc_out)
        total_params += self.fc_1.weight.numel() + self.fc_1.bias.numel()
        total_params += self.fc_out.weight.numel() + self.fc_out.bias.numel()

        return total_params

    def _initialize_weights(self):
            # Initialize weights for rnn_real
            for name, param in self.RNN.named_parameters():
                if 'weight_ih' in name:  # Input-hidden weights
                    nn.init.eye_(param)  # Initialize to identity matrix
                elif 'weight_hh' in name:  # Hidden-hidden weights
                    nn.init.orthogonal_(param, gain=nn.init.calculate_gain('relu'))  # Orthogonal initialization
                elif 'bias' in name:
                    nn.init.zeros_(param)  # Bias to zero


                    # Initialize the output layer (Linear layer)
            nn.init.kaiming_uniform_( self.fc_1.weight,nonlinearity='relu')  # Weights set to one
            nn.init.zeros_(self.fc_1.bias)  # Bias to zero
            nn.init.kaiming_uniform_(self.fc_out.weight,nonlinearity='relu')  # Weights set to one
            nn.init.zeros_(self.fc_out.bias)  # Bias to zero

    def forward(self, sig,h_0,c_0):

        res=sig
        sig = sig.view(self.batch_sz,self.seqL,self.input_size)
        if self.rnn_type == 'LSTM':
            out,(h_nxt,c_nxt) = self.RNN(sig,(h_0,c_0))
        else:
            out,h_nxt = self.RNN(sig,h_0)
        FC_input = out.flatten(start_dim=1)
        FC_input = self.bn_1(FC_input)
        mid = self.fc_1(FC_input)
        final = self.bn_2(self.relu(mid))
        sig_out = self.fc_out(final) + res
        if self.rnn_type=='LSTM':
            return sig_out ,h_nxt,c_nxt
        else:
            return sig_out, h_nxt



