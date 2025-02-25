
import torch.nn.init as init
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import matplotlib
import scipy.signal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MATLABDataset_spectogram(Dataset):
    def __init__(self, mat_file, signal_len=80000,split_ratio=0.8,fco_distor=False
                 ,n_fft=1024,hop_len=256, mode="spectrogram"):
        """
        Args:
            mat_file (str): Path to the .mat file.
            signal_len (int): Desired constant length for all signals.
        """

        mat_data = scipy.io.loadmat(mat_file)
        self.sig_len = signal_len
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.fco_distor = fco_distor
        self.clean_signals = [sig.flatten() for sig in mat_data['IQ_dataset_param']['clean_sig'][0][0][0]]
        self.distor_signals = [sig.flatten() for sig in mat_data['IQ_dataset_param']['distor_sig'][0][0][0]]

        # Split into train and validation sets
        split_idx = int(len(self.clean_signals) * split_ratio)
        self.train_clean_signals = self.clean_signals[:split_idx]
        self.train_distor_signals = self.distor_signals[:split_idx]
        self.val_clean_signals = self.clean_signals[split_idx:]
        self.val_distor_signals = self.distor_signals[split_idx:]
        # self.train_clean_spectrograms = [self.gen_spectogram(sig) for sig in
        #                                  tqdm(self.train_clean_signals, desc="Processing train clean signals")]
        # self.train_distor_spectrograms = [self.gen_spectogram(sig) for sig in
        #                                   tqdm(self.train_distor_signals, desc="Processing train distor signals")]
        # self.val_clean_spectrograms = [self.gen_spectogram(sig) for sig in
        #                                tqdm(self.val_clean_signals, desc="Processing val clean signals")]
        # self.val_distor_spectrograms = [self.gen_spectogram(sig) for sig in
        #                                 tqdm(self.val_distor_signals, desc="Processing val distor signals")]
        self.fs = 20*26*(10**8)
        self.fc = self.fs / 20
        self.mode = mode
        #self.gen_spectogram(self.clean_signals[1],self.distor_signals[1])


    def gen_spectogram(self,sig ):
        sig = torch.tensor(sig, dtype=torch.complex64)
        sig = sig.view(1, self.sig_len)
        sig = sig.reshape(self.sig_len)
        window = torch.hann_window(self.n_fft)
        stft_output = torch.stft(sig, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.n_fft,
                                 window=window, return_complex=True, normalized=True)

        return stft_output

    def set_train(self):
        """
        Set the dataset to training mode.
        """
        self.clean_sig = self.train_clean_signals
        self.distor_sig =  self.train_distor_signals

    def rtn_spectogram_param(self):
        return self.n_fft,self.hop_len
    def set_validation(self):
        """
        Set the dataset to validation mode.
        """
        self.clean_sig = self.val_clean_signals
        self.distor_sig = self.val_distor_signals

    def get_train_set(self):
        """
        Returns the training dataset in the format expected by the network.
        """
        return [self[i] for i in range(len(self.train_clean_signals))]

    def get_validation_set(self):
        """
        Returns the validation dataset in the format expected by the network.
        """
        return [self[i] for i in range(len(self.val_clean_signals))]

    def __len__(self):
        return len(self.clean_signals)

    def set_mode(self, mode):
        """Set mode to either 'spectrogram' or 'original+spectrogram'."""
        assert mode in ["spectrogram", "original+spectrogram"], "Invalid mode"
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "spectrogram":
            # Process spectrogram as usual

            stft_clean_real = np.real(self.gen_spectogram(self._scale_complex(self.clean_sig[idx])))
            stft_clean_imag = np.imag(self.gen_spectogram(self._scale_complex(self.clean_sig[idx])))
            stft_distor_real = np.real(self.gen_spectogram(self._scale_complex(self.distor_sig[idx])))
            stft_distor_imag = np.imag(self.gen_spectogram(self._scale_complex(self.distor_sig[idx])))
            return stft_clean_real, stft_clean_imag, stft_distor_real, stft_distor_imag

        elif self.mode == "original+spectrogram":
            # Return original signals and spectrograms
            original_clean = self._scale_complex(self.clean_sig[idx])
            original_distor = self._scale_complex(self.distor_sig[idx])

            stft_clean_real = np.real(self.gen_spectogram(self._scale_complex(self.clean_sig[idx])))
            stft_clean_imag = np.imag(self.gen_spectogram(self._scale_complex(self.clean_sig[idx])))
            stft_distor_real = np.real(self.gen_spectogram(self._scale_complex(self.distor_sig[idx])))
            stft_distor_imag = np.imag(self.gen_spectogram(self._scale_complex(self.distor_sig[idx])))

            return original_clean, original_distor, stft_clean_real, stft_clean_imag, stft_distor_real, stft_distor_imag


    def _scale_complex(self,sig):
        min = abs(sig).min()
        max = abs(sig).max()
        norm_01 = (sig - min) / (max - min)
        norm_100 = norm_01 * 2 - 1
        return norm_100

class MATLABDataset_inter(Dataset):
    def __init__(self, mat_file, signal_len,split_ratio=0.8,fco_distor=False):
        """
        Args:
            mat_file (str): Path to the .mat file.
            signal_len (int): Desired constant length for all signals.
        """
        mat_data = scipy.io.loadmat(mat_file)
        self.fco_distor = fco_distor
        self.clean_signals = [sig.flatten() for sig in mat_data['IQ_dataset_param']['clean_sig'][0][0][0]]
        self.distor_signals = [sig.flatten() for sig in mat_data['IQ_dataset_param']['distor_sig'][0][0][0]]

        # Split into train and validation sets
        split_idx = int(len(self.clean_signals) * split_ratio)
        self.train_clean_signals = self.clean_signals[:split_idx]
        self.train_distor_signals = self.distor_signals[:split_idx]
        if self.fco_distor:
            self.fco_tar = [sig.flatten() for sig in mat_data['IQ_dataset_param']['fco'][0][0][0]]
            self.train_fco = self.fco_tar[:split_idx]
            self.val_fco = self.fco_tar[split_idx:]

        self.val_clean_signals = self.clean_signals[split_idx:]
        self.val_distor_signals = self.distor_signals[split_idx:]


    def _truncate_or_pad(self, signal, target_len):
        """
        Truncates or pads the signal to the target length.
        """
        if len(signal) > target_len:
            return signal[:target_len]  # Truncate
        elif len(signal) < target_len:
            return np.pad(signal, (0, target_len - len(signal)), 'constant')  # Pad
        return signal

    def set_train(self):
        """
        Set the dataset to training mode.
        """
        self.clean_signals = self.train_clean_signals
        self.distor_signals = self.train_distor_signals

    def set_validation(self):
        """
        Set the dataset to validation mode.
        """
        self.clean_signals = self.val_clean_signals
        self.distor_signals = self.val_distor_signals

    def get_train_set(self):
        """
        Returns the training dataset in the format expected by the network.
        """
        return [self[i] for i in range(len(self.train_clean_signals))]

    def get_validation_set(self):
        """
        Returns the validation dataset in the format expected by the network.
        """
        return [self[i] for i in range(len(self.val_clean_signals))]


    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        # Get real and imaginary parts of clean and distorted signals
        clean_signal = self.clean_signals[idx]
        distor_signal = self.distor_signals[idx]

        clean_real = self._scale(torch.tensor(np.real(clean_signal), dtype=torch.float32))
        clean_imag = self._scale(torch.tensor(np.imag(clean_signal), dtype=torch.float32))
        distor_real = self._scale(torch.tensor(np.real(distor_signal), dtype=torch.float32))
        distor_imag = self._scale(torch.tensor(np.imag(distor_signal), dtype=torch.float32))

        clean_sig = self._interleave(clean_real,clean_imag)
        distor_sig = self._interleave(distor_real, distor_imag)
        if self.fco_distor:
            fco_tar_inKhz = torch.tensor(self.fco_tar[idx],dtype=torch.float32) / 10e3
            return clean_sig,distor_sig,fco_tar_inKhz
        # Pack in a way that matches what the network expects
        return clean_sig,distor_sig

    def _interleave(self,real_sig,imag_sig):
        interleaved = torch.empty(2 *  real_sig.size(0), dtype=real_sig.dtype)
        interleaved[0::2] = real_sig
        interleaved[1::2] = imag_sig
        return interleaved

    def _scale(self,sig):
        min = sig.min()
        max = sig.max()
        norm_01 = (sig - min) / (max - min)
        norm_100 = norm_01 * 2 - 1
        return norm_100

class MATLABDataset(Dataset):
    def __init__(self, mat_file, signal_len):
        """
        Args:
            mat_file (str): Path to the .mat file.
            signal_len (int): Desired constant length for all signals.
        """
        mat_data = scipy.io.loadmat(mat_file)

        self.clean_signals = [sig.flatten() for sig in mat_data['IQ_dataset_param']['clean_sig'][0][0][0]]
        self.distor_signals = [sig.flatten() for sig in mat_data['IQ_dataset_param']['distor_sig'][0][0][0]]

    def _truncate_or_pad(self, signal, target_len):
        """
        Truncates or pads the signal to the target length.
        """
        if len(signal) > target_len:
            return signal[:target_len]  # Truncate
        elif len(signal) < target_len:
            return np.pad(signal, (0, target_len - len(signal)), 'constant')  # Pad
        return signal

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        # Get real and imaginary parts of clean and distorted signals
        clean_signal = self.clean_signals[idx]
        distor_signal = self.distor_signals[idx]

        clean_real = self._scale(torch.tensor(np.real(clean_signal), dtype=torch.float32))
        clean_imag = self._scale(torch.tensor(np.imag(clean_signal), dtype=torch.float32))
        distor_real = self._scale(torch.tensor(np.real(distor_signal), dtype=torch.float32))
        distor_imag = self._scale(torch.tensor(np.imag(distor_signal), dtype=torch.float32))

        # Pack in a way that matches what the network expects
        return distor_real, distor_imag, clean_real, clean_imag
    def _scale(self,sig):
        min = sig.min()
        max = sig.max()
        norm_01 = (sig - min) / (max - min)
        norm_100 = norm_01 * 2 - 1
        return norm_100

 #


 # def _scalecomplex(self, sig):
    #     abs_sig = abs(sig)
    #     min = sig.min(abs_sig)
    #     max = sig.max(abs_sig)
    #     norm_01 = (abs_sig - min) / (max - min)
    #     norm = norm_01 * 2 - 1
    #     sig_out = sig / norm
    #     return sig_out