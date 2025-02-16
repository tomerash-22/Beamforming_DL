function [rx_ofdm_sig,rx_ofdm_tar,rx_clean] = ...
rx_ofdm(sig_param , impairments_param,pre_mod_sig,clean_premod)
pre_mod_sig=pre_mod_sig';
clean_premod=clean_premod';
bin_amu_sig= round((length(pre_mod_sig(1,:))/2)* ...
    (sig_param.ofdm_BW/(sig_param.fs/2)));
fc_base_band=round(length(pre_mod_sig(1,:))/2);
sig_loc=1:bin_amu_sig;
rect_lp = zeros(size(pre_mod_sig(1,:)));
rect_lp(sig_loc) = ones(size(sig_loc)) ;
rect_lp(end-fliplr(sig_loc)) = ones(size(sig_loc));
%% Baseband OFDM modulator based QPSK
FFTlen=128;
%groups = N/FFTlen ; 
del_f = sig_param.ofdm_BW / FFTlen; %1 FFT bin freq content
N_bins = round(sig_param.fs/del_f); 
Q_samp=zeros(size(pre_mod_sig));
I_samp=zeros(size(pre_mod_sig));
ts=1/sig_param.fs;
t=(0:ts:ts*(N_bins-1));
%IQ imblance I-sin
% for sen=1:4
%     %delted-2pi
% I_samp(sen,:) =impairments_param.gain(1,sen)*...
% cos(2*pi*(sig_param.fc+sig_param.ofdm_BW/2).*t ) ; 
% Q_samp(sen,:) =  sin(2*pi*((sig_param.fc+sig_param.ofdm_BW/2).*t)+ impairments_param.phase(1,sen)) ;
% 
% end
% I_samp_tar =repmat( cos(2*pi*((sig_param.fc+sig_param.ofdm_BW/2).*t)) ,4,1);
% Q_samp_tar=repmat(sin(2*pi*((sig_param.fc+sig_param.ofdm_BW/2).*t)) ,4,1);

% for sen=1:4
%     %delted-2pi
% I_samp(sen,:) =db2mag(0.5)*...%impairments_param.gain(1,sen)*...
% cos(2*pi*(sig_param.fc).*t ) ; 
% Q_samp(sen,:) =  sin(2*pi*((sig_param.fc).*t)+deg2rad(4)) ;    %impairments_param.phase(1,sen)) ;
% end

rx_ofdm_sig = zeros(size(pre_mod_sig));
rx_clean=zeros(size(rx_ofdm_sig));
rx_ofdm_tar=zeros(size(rx_clean));
%fco
% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(inter_sig)),db(abs(fftshift(fft((inter_sig_t2)))), 200));
% hold on
% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(inter_sig)),db(abs(fftshift(fft((sig)))), 200));

for sen=1:4
  rx_ofdm_sig(sen,:)=LP_and_comb(sig_param,impairments_param, ...
      pre_mod_sig,sen,rect_lp,true);  
  rx_ofdm_tar(sen,:)=LP_and_comb(sig_param,impairments_param, ...
      pre_mod_sig,sen,rect_lp,false);
  rx_clean(sen,:)=LP_and_comb(sig_param,impairments_param, ...
      clean_premod,sen,rect_lp,true);

end
% 
%    figure(2)
%  plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(rx_ofdm_sig(1,:))), ...
%      db(abs(fftshift(fft((rx_ofdm_sig(1,:))))), 200));
% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(pre_mod_sig(1,:))), ...
%     db(abs(fftshift(fft((rx_ofdm_tar(1,:))))), 200));
%  plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(pre_mod_sig(1,:))), ...
%      db(abs(fftshift(fft((pre_mod_sig(1,:))))), 200));

end

function lp_sig = LP_and_comb (sig_param,impairments_param,sig,sen,rect_lp, ...
    IQdistor)

FFTlen=128;
%groups = N/FFTlen ; 
del_f = sig_param.ofdm_BW / FFTlen; %1 FFT bin freq content
N_bins = round(sig_param.fs/del_f); 
ts=1/sig_param.fs;
t=(0:ts:ts*(N_bins-1));

I_samp = cos(2*pi*((sig_param.fc).*t));
Q_samp =sin(2*pi*((sig_param.fc).*t));

% sig_Q = real(sig(sen,:));
% sig_I = imag(sig(sen,:));
sig_Q= sig(sen,:).*Q_samp;
sig_I = sig(sen,:).*I_samp;
%debug only-
% unfilt_sig = sig_Q - 1j.*sig_I;
% fs=2.6e9 * 20;

% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(pre_mod_sig(1,:))), ...
%     db(abs(fftshift(fft((pre_mod_sig(1,:))))), 200));


if(IQdistor)
    sig_Q = sig_Q.*exp(1j*deg2rad(impairments_param.phase(1,sen)));
    sig_I=sig_I.*db2mag(impairments_param.gain(1,sen));
end
unfilt_sig =sig_I + 1j*sig_Q;
plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(unfilt_sig)), ...
    db(abs(fftshift(fft((unfilt_sig)))), 200));
sig_Q = ifft (fft(sig_Q).*rect_lp);
sig_I = ifft( fft(sig_I).*rect_lp);
lp_sig =sig_I + 1j*sig_Q;
%test_IQ(lp_sig,sig_param,impairments_param,true );
end


%% retry - works after copying yuval
function test = test_IQ (sig ,sig_param,impairments_param,distor)
I_sig = imag(sig); 
Q_sig= real(sig);
beta_I = mean(I_sig);
beta_Q = mean(Q_sig);
I_sig = I_sig - beta_I;
Q_sig = Q_sig - beta_Q;
%normlize
I_sig = I_sig/std(Q_sig)/sqrt(2);
Q_sig = Q_sig/std(Q_sig)/sqrt(2);

II_inner =  mean(I_sig.*I_sig);
IQ_inner = mean(I_sig.*Q_sig);

alpha = sqrt (2*II_inner);
sin_phi =  (2/alpha)*IQ_inner;
cos_phi = sqrt(1-(sin_phi^2));
A=1/alpha;
C=-1*sin_phi/(alpha*cos_phi);
D=1/cos_phi;

corr_mat= [A , 0 ; C ,D];
iqmat= corr_mat*[I_sig ;Q_sig]; 
Q_corr=iqmat(1,:);
I_corr=iqmat(1,:);
gain_aproxx=mag2db(A);
phase_aproxx=asin(sin_phi)*(180/pi); %D~exp(1j*phi) phi = imag(log(D))

a=db2mag(1.5);


end

%% add impair to lp_andcomb - i.e. add impairments in BB not in fc down conversion
%% change phase to match yuval
%% change parametrs 
