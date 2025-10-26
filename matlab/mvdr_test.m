
clear all
close all

[ch_param, impairments_param, dataset_param_coupling,...
    IQ_dataset_param, sim_param, sig_param,...
    inter_param,snr_gain,signal_data] = init_params();

ULA_0AZ_24_09 = importdata("ula_2_6Ghz_7_11.mat");

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


[signal_data] = sig_gen_ofdm...
    (signal_data ,sig_param , inter_param, ch_param ,ULA_0AZ_24_09,impairments_param );


 
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

ii=ii+1

end









