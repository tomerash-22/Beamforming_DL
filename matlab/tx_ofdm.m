function [tx_ofdm_sig] = ...
tx_ofdm(sig_param , impairments_param)

%% Baseband OFDM modulator based QPSK

fs=sig_param.fs;
N=6.4*10e5;
FFTlen=128;
%groups = N/FFTlen ; 
del_f = sig_param.ofdm_BW / FFTlen; %1 FFT bin freq content
N_bins = round(fs/del_f); 
del_fc_bin = (sig_param.fc/fs); 
bw_bin = sig_param.ofdm_BW/del_f;
%X_st = zeros(1,N_bins);

bits_per_symb=2;
BIT_stream_len = FFTlen*bits_per_symb; 
I_st = zeros(1,BIT_stream_len/bits_per_symb);
Q_st = zeros(1,BIT_stream_len/bits_per_symb);
bit_stream = sign(randn(1,BIT_stream_len));
ts=1/fs;
tc=1/sig_param.fc;
%t = 0:tc:tc*(N_bins-1);
%+(sig_param.ofdm_BW/2)
t=(0:ts:ts*(N_bins-1));

 Q_samp = sin(2*pi*((sig_param.fc).*t));
 I_samp = cos(2*pi*((sig_param.fc).*t));
for k=1:1:BIT_stream_len/2
    I_st(1,k)=bit_stream(1,2*k-1);
    Q_st(1,k)=bit_stream(1,2*k);
end

n_bin=1:1:N_bins;
X_st = I_st + 1j.*Q_st; %QPSK mapper
x_st_t = (ifft(X_st,N_bins)); %stil digital symbol
Q_sig = real(x_st_t);
I_sig = imag(x_st_t);
x_st_t = (I_sig.*I_samp)-(Q_sig.*Q_samp);
%x_st_t = (x_st_t.*(Q_samp-I_samp));
tx_ofdm_sig=x_st_t;

%% plot
%plot(linspace(-fs/2,fs/2,length(tx_ofdm_sig)),db(abs(fftshift(fft((tx_ofdm_sig)))), 200));

% 
% Q_samp =sqrt(2)*cos(2*pi*((sig_param.fc+sig_param.ofdm_BW/2).*t));
% I_samp =sqrt(2)* sin(2*pi*((sig_param.fc+sig_param.ofdm_BW/2).*t));







