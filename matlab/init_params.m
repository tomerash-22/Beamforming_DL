function [ch_param, impairments_param, dataset_param_coupling,...
    IQ_dataset_param, sim_param, sig_param, inter_param,snr_gain,signal_data] = init_params()
    samp_pt=10e5;
fk=1e6 ; %1Mhz bin for OFDM

ch_param.snr = 13;
ch_param.snapshot_amu = 500;
ch_param.mvdr_mpdr = 'MVDR' ; 

impairments_param.Z_mat = zeros(4,4);
%IQ imblance param. 
impairments_param.phase=0;
impairments_param.gain=0;
impairments_param.fco=0;


dataset_param_coupling.amu = 4;
dataset_param_coupling.Z_mat = zeros(4,4,dataset_param_coupling.amu);
dataset_param_coupling.R_sig = zeros(4,4,dataset_param_coupling.amu);
dataset_param_coupling.R_coupled = zeros(4,4,dataset_param_coupling.amu);

% Initialize a struct with a cell array field
IQ_dataset_param = struct();
IQ_dataset_param.clean_sig = {};  % Empty cell array for clean signals
IQ_dataset_param.distor_sig = {};  % Empty cell array for distorted signals
IQ_dataset_param.fco ={};

snr_gain.mvdr=0;
snr_gain.mpdr=0;
snr_gain.matlab=0;
snr_gain.mvdr_fl=0;
snr_gain.matlab_fl=0;
snr_gain.snrgain_th=5.5;

sim_param.BF_th=4.5;
sim_param.lambada_step=1e-9; 

sig_param.fc=26e8;
sig_param.fsig=26; % in fs
sig_param.inputAngle= [33;0];
%sig_param.t=(0:ts:7999*ts).';
sig_param.dir=33;
sig_param.lamda = 3e8 / sig_param.fc;
sig_param.fk=fk;
sig_param.ofdm_BW=20e6;
sig_param.fs=20*sig_param.fc;
sig_param.snr=ch_param.snr;

inter_param.corel_fact =0.00;
inter_param.inputAngle=[-5;0];
inter_param.dir = -5;
inter_param.attindB = -20;
inter_param.del_f = 3.77e5; %signed frequency diffrence
inter_param.BW=5e6;
inter_param.exict = false;
inter_param.fs = sig_param.fs;
inter_param.fc=sig_param.fc;
inter_param.snr=sig_param.snr;




% Initialize the struct to hold all signals in cells
signal_data = struct('rx_sig', [], 'rx_sig_ofdm', [], 'rx_sig_lowsnr', [], ...
    'rx_clean_sig', [], 'R_inter_noise', [], 'rx_tar', [], 'doa', [], ...
     'R_sig',[]);




end
