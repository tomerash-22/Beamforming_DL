

clear all
close all

[ch_param, impairments_param, dataset_param_coupling,...
    IQ_dataset_param, sim_param, sig_param,...
    inter_param,snr_gain,signal_data] = init_params();

ULA_0AZ_24_09 = importdata("ula_2_6Ghz_7_11.mat");
% Txsig = load("ofdm_14_10.mat");
% filt_320 = load("320_bub.mat");

[ch_param.collector_nw,ch_param.collector_w,ch_param.estm,...
    ch_param.bf,ch_param.transmitter,ch_param.ofdm_transmitter...
    ,ch_param.ofdm_rx] =config(sig_param  , ULA_0AZ_24_09);
ch_param.sv = zeros(1,4)';
ii=1;
failed_cnt =0;
failed_cnt_mvdr=0;

del_ang = zeros(1,100);
sir = zeros(1,100);
snr_f = zeros(1,100);
ang_sig= zeros(1,100);
inter_sig= zeros(1,100);
bw_ofdm=zeros(1,100);

while (ii <= dataset_param_coupling.amu) 

[sig_param,inter_param,ch_param,impairments_param]=...
Randomizesim_param(sig_param,inter_param,ch_param,impairments_param);
%debug only
%  ch_param.snr=100;
%  inter_param.attindB=100;


[signal_data] = sig_gen_ofdm...
    (signal_data ,sig_param , inter_param, ch_param ,ULA_0AZ_24_09,impairments_param );

%doa
% release(ch_param.estm);
% ch_param.estm.OperatingFrequency = 2.6e9; 
%     [~,doa] = ch_param.estm(rx_sig);
 
[w,w_mvdr,signal_data,ch_param,snr_gain]=MVDR(signal_data, ...
     sig_param,ch_param,ULA_0AZ_24_09  ...
    ,inter_param,snr_gain);
try
%check the stats for failed BF
if(BF_failed(sig_param,inter_param,ULA_0AZ_24_09,w_mvdr,w,snr_gain) )
    failed_cnt_mvdr = failed_cnt_mvdr+1;
    del_ang(1,failed_cnt_mvdr) = abs(sig_param.dir - inter_param.dir);
    sir(1,failed_cnt_mvdr) = inter_param.attindB; 
    snr_f(1,failed_cnt_mvdr) =ch_param.snr;
    ang_sig(1,failed_cnt_mvdr) = sig_param.dir;
    inter_sig(1,failed_cnt_mvdr) = inter_param.dir;
    bw_ofdm(1,failed_cnt_mvdr) = sig_param.ofdm_BW;
    continue
end
catch 
    continue
end

sig_len = 80000;
 IQ_dataset_param = add_impair(signal_data,impairments_param,IQ_dataset_param ...
     ,ii,'FCO', sig_len);
% dataset_param_coupling= add_impair(signal_data,impairments_param,dataset_param_coupling ...
%     ,ii,'coupling', sig_len);
ii=ii+1

% if (mod(ii,10)==0)
% figure(2)
% pattern(ULA_0AZ_24_09,sig_param.fc,-90:90,0, ...
%     'Weights',w,'Type','directivity',...
%     'PropagationSpeed',physconst('LightSpeed'),...
%     'CoordinateSystem','rectangular'); 
% end
end

%% Q 
% 1. compare two approches for rx - sample delay vs steering vector  -
% steering vec

% 2. should i put more effort into making good w vector? (diagonal loading)
% - increse to 5.5 snr gain.
% 3. is w has a norm limit?

% 4. should the IQ imblance work with constant (trimmed) signals? or not
 
% 5. discuss regulaztion changes in coupling CNN (A-A^H)
% 6.note that ive decresed ofdm BW and white noise BW w.r.t smaller fs and
% - increse BW to 20Mhz
% fc
% 7. discuss training in the computer labs
% 8. lab admin - mail to nimrod peleg about 1 man proj
% mabey use RNN instead of LSTM . 

%% A
%1. take const length at start (10,000 samples ? ) 
% gain - +-0.5-1.5 [db] , phase 5-10 [deg] ;
%ensure msymterical signal around zero 
% om@technion.ac.il







