function [signal_data] = ...
sig_gen_ofdm(signal_data,sig_param , inter_param...
    , ch_param, ULA_0AZ_24_09 ,impairments_param)


sig =  tx_ofdm(sig_param,impairments_param);
del_bin = sig_param.fs/length(sig);
bin_amu= round((length(sig)/2)*(inter_param.BW/(sig_param.fs/2)));
bin_amu_sig= round((length(sig)/2)*(sig_param.ofdm_BW/(sig_param.fs/2)));
fc_bin=round((length(sig)/2)*(sig_param.fc/(sig_param.fs/2)));
noise_loc=fc_bin-round(bin_amu/2):fc_bin+round(bin_amu/2);
sig_loc=fc_bin-round(bin_amu_sig/2):fc_bin+round(bin_amu_sig/2);


f_sig = fft(sig);
sig_lv=(mean(db(abs(f_sig(sig_loc)))));
noise = wgn(length(sig),1,-30)';
f_noise = fft(noise);
noise_lv = (mean(db(abs(f_noise))));
snr_lv = sig_lv-noise_lv;
inter_diff = inter_param.attindB - snr_lv ;  
snr_diff = ch_param.snr-snr_lv;
snr_low = 10-snr_lv;
f_inter = f_noise.*db2mag(-1*inter_diff);
f_noise = f_noise.*db2mag(-1*snr_diff);
f_low_noise=f_noise.*db2mag(-1*snr_low);

noise_snr=ifft(f_noise).*2;
low_noise = ifft(f_low_noise).*2;
rect=zeros(size(f_inter));
rect(noise_loc)=ones(size(noise_loc));
f_inter =f_inter.*rect;
f_inter =(1/sqrt(2)).*(fliplr(f_inter)+f_inter);
inter_sig_t2=ifft(f_inter);

rset_sys(ch_param);

    signal_data.rx_clean_sig{1} = rx_func(sig,sig_param,ULA_0AZ_24_09,ones(4,1));     
    rx_inter =rx_func(inter_sig_t2  ,inter_param,ULA_0AZ_24_09,ones(4,1));
    rx_noise = repmat(noise_snr,[4,1])';
    rx_lownoise= repmat(low_noise,[4,1])';
    signal_data.rx_sig{1} = rx_noise+rx_inter+signal_data.rx_clean_sig{1};

% 
[signal_data.rx_sig_ofdm{1},~,signal_data.rx_tar{1}] =...
rx_ofdm(sig_param , impairments_param,signal_data.rx_sig{1},signal_data.rx_clean_sig{1});

    

% if ( ch_param.snr>11 || ch_param.snr==0 ) %
%     rset_sys(ch_param); 
%     rx_sig_lowsnr =rx_lownoise+rx_inter+rx_clean_sig;
% else
%     rx_sig_lowsnr = rx_sig;
% end

signal_data.rx_sig_lowsnr{1} = signal_data.rx_sig{1};
signal_data.R_inter_noise{1} = cov(  signal_data.rx_sig{1} -signal_data.rx_clean_sig{1} );


%% diffrent imple : 
%      rx_clean_sig = ch_param.collector_nw(sig',sig_param.inputAngle);     
%      rx_inter =ch_param.collector_nw(inter_sig_t2',inter_param.inputAngle);
%      rx_noise = repmat(noise_snr,[4,1]);

    %rx_sig = awgn(rx_clean_sig,ch_param.snr)+rx_inter;
%% FFT
% figure(2)
% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(sig)), ...
%     db(abs(fftshift(fft((rx_sig(:,1))))), 200));
% hold on
% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(sig)), ...
%     db(abs(fftshift(fft((rx_clean_sig(:,1))))), 200));
% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(sig)), ...
%     db(abs(fftshift(fft((rx_inter(:,1))))), 200));
% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(sig)), ...
%     db(abs(fftshift(fft((rx_noise(:,1))))), 200));

