
import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
from Matlab_dataloaders import MATLABDataset_spectogram
def initialize_weights(m):
    """Apply weight initialization to Conv2D and Conv1D layers."""
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)  # Xavier initialization for conv layers
        if m.bias is not None:
            init.zeros_(m.bias)

class conv1D_out(nn.Module):
    def __init__(self, in_channels,kernel_sz=(1,3),padding=(0,1)):
        super(conv1D_out, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, 1,kernel_sz, padding='same')

    def forward(self, x):
        out = self.conv1d(x)
        return out


class conv1D(nn.Module):
    def __init__(self, in_channels,kernel_sz=(1,3),padding=(0,1)):
        super(conv1D, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, in_channels,kernel_sz, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)  # Normalizing across the batch dimension
        return x


class conv2D(nn.Module):
    def __init__(self, in_channels, out_channels,dial_freq, kernel_size=(9,3), kernel_1d=(1,3)):
        super(conv2D, self).__init__()
        if dial_freq==1:
            self.res_conn=False
        else:
            self.res_conn = True

        self.padding = (4,dial_freq)
        self.freq_dialted_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                           padding=self.padding,dilation=(1,dial_freq))
        self.conv1d = nn.Conv2d(out_channels, out_channels, kernel_1d, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.res_conn:
            res = x
        x = self.freq_dialted_conv(x)
        x = self.relu(x)
        x=self.conv1d(x)
        if self.res_conn:
            x = x + res
        return x


class IQ_spectogram_CNN_res_skipconn(nn.Module):
    def __init__(self, kernel2d_size=(9,3),kernel_1d_size=(1,3),channels=48,dep=4):
        super(IQ_spectogram_CNN_res_skipconn, self).__init__()

        self.conv2D_layers = nn.ModuleList([
        conv2D(in_channels=1 if i==0 else channels,
               out_channels=channels,
               dial_freq=2 ** i,
               kernel_size=kernel2d_size,
               kernel_1d=kernel_1d_size) for i in range(dep)])
        self.conv1D = conv1D(in_channels=channels*dep,kernel_sz=kernel_1d_size)
        self.convout=conv1D_out(in_channels=channels*dep,kernel_sz=kernel_1d_size)
        initialize_weights(self.convout),initialize_weights(self.conv1D)
        for layer in self.conv2D_layers:
            initialize_weights(layer)

    def forward(self, sepctogram_in):
        to1dconv = []
        #sepctogram_in=sepctogram_in.unsqueeze(1)
        idx = 0
        for layer in self.conv2D_layers:
            if idx == 0:
                temp = layer(sepctogram_in)
            else:
                temp = layer(temp)
            to1dconv.append(temp)
            idx += 1
        to1dconv = torch.cat(to1dconv,dim=1)
        toout1dconv = self.conv1D(to1dconv)
        out = self.convout(toout1dconv)
        return out

def test_reconstract(model,sig_len=80000):
     model.eval()
     with ((torch.no_grad())):
         val_loss = 0.0
         diff_val = 0.0
         criterion = nn.MSELoss()
         dataset = MATLABDataset_spectogram('IQ_400sig_10_02.mat', signal_len=sig_len)
         dataset.set_mode("original+spectrogram")
         dataset.set_validation()
         val_dataset = dataset.get_validation_set()
         val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
         n_fft,hop_len = dataset.rtn_spectogram_param()

         for (original_clean, original_distor, stft_clean_real,
              stft_clean_imag, stft_distor_real, stft_distor_imag) in val_loader:

            window = torch.hann_window(n_fft)
            stft_complex = stft_distor_real + 1j * stft_distor_imag
            reconstructed_distor = torch.istft(stft_complex, n_fft=n_fft, hop_length=hop_len, win_length=n_fft,
                                               window=window,return_complex=True,center=True,normalized=True)

            recon_real = np.real(reconstructed_distor)
            recon_imag = np.imag(reconstructed_distor)
            original_real,original_imag = np.real(original_clean),np.real(original_distor)
            diff_val += criterion(recon_real,original_real[:,:len(reconstructed_distor[0])]) \
                        + criterion(recon_imag, original_imag[:, :len(reconstructed_distor[0])])



        # diff = torch.sum((sig - reconstructed_signal)**2)











        # # spectrogram = torch.log1p(nn.functional.normalize(input=torch.abs(stft_output)))
        # # plt.figure(figsize=(10, 5))
        # # plt.imshow(spectrogram.numpy(), aspect="auto", origin="lower", cmap="magma")
        # # plt.colorbar(label="Magnitude (log scale)")
        # # plt.xlabel("Time Frames")
        # # plt.ylabel("Frequency Bins")
        # # plt.title("STFT Spectrogram")
        # # plt.show()
        #
        # # Spctogram_out,_,_,_ = plt.specgram(x=sig, NFFT=self.n_fft, Fs=self.fs,
        # #                                    Fc=self.fc, noverlap=int(self.hop_len / 2))
        # # phase= np.angle(Spctogram_out)
