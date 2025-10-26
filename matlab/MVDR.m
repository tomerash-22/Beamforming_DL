function [w,w_mvdr,signal_data,ch_param,snr_gain] = ...
MVDR(signal_data, ...
    sig_param ,ch_param , ULA_0AZ_24_09...
    ,inter_param,snr_gain )

    sig_est_dir=sig_param.dir;


signal_data.R_sig{1} = cov(signal_data.rx_sig{1});
figure(4)

%steerring vector
%in our case d=lamda/2

f_st = 0.5*sin( deg2rad(sig_est_dir));
a_sv = [1 exp(1j*2*pi*f_st) exp(1j*4*pi*f_st) exp(1j*6*pi*f_st) ]';

if (sig_param.dir>0)
    a_sv =flipud(a_sv);
end
asv_H = a_sv';

% wieghts
%w = (R_sig\a_sv) / (asv_H*(R_sig\a_sv));
release(ch_param.bf);
ch_param.bf.TrainingInputPort=true;
%train on noise+interfirence then act on Rx
[~,w] = ch_param.bf(signal_data.rx_sig{1}, ...
    signal_data.rx_sig_lowsnr{1}-signal_data.rx_clean_sig{1} ...
    , [sig_est_dir;0]);
w_mvdr = (signal_data.R_inter_noise{1}\a_sv) / (asv_H*(signal_data.R_inter_noise{1}\a_sv));
reset(ch_param.bf);

% S.C. steps:
 %plot pattern sainity check in high snr
% calculate snr via radition pattern
% snr_gain.mvdr= snr_calc(w_mvdr,doa,ULA_0AZ_24_09,sig_param, ...
%     inter_param,'MVDR');
% snr_gain.mpdr = snr_calc(w,doa,ULA_0AZ_24_09,sig_param, ...
%     inter_param,'MPDR');


try
    snr_gain.matlab = snr_calc(w,doa,ULA_0AZ_24_09,sig_param, ...
    inter_param,'MATLAB');
    snr_gain.mvdr = snr_calc(w_mvdr,doa,ULA_0AZ_24_09,sig_param, ...
    inter_param,'MATLAB');
catch
    warning('doa problem?')
    snr_gain.matlab = 0 ;
end











