
import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


    def _scale(self, sig):
        min = sig.min()
        max = sig.max()
        norm_01 = (sig - min) / (max - min)
        norm_100 = norm_01 * 200 - 100
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

    def _scale(self, sig):
        min = sig.min()
        max = sig.max()
        norm_01 = (sig - min) / (max - min)
        norm_100 = norm_01 * 200 - 100
        return norm_100

