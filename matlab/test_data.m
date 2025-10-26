
sig_param.fc=26e8;
sig_param.fsig=26; % in fs
sig_param.inputAngle= [33;0];
%sig_param.t=(0:ts:7999*ts).';
sig_param.dir=33;
sig_param.lamda = 3e8 / sig_param.fc;

sig_param.ofdm_BW=20e6;
sig_param.fs=20*sig_param.fc;


data = load('full_signal_examples_complex.mat');
input_signals = data.input_signals;
output_signals = data.output_signals;
target_signals = data.target_signals;

sig_input = squeeze(input_signals(1,1,:)) ;
sig_output =  output_signals(1,1,:);
sig_target =  target_signals(1,1,:);

% plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(sig)), ...
%     db(abs(fftshift(fft((rx_clean_sig(:,1))))), 200));

plot(linspace(-sig_param.fs/2,sig_param.fs/2,length(sig_input)), ...
    db(abs(fftshift(fft(sig_input))), 200));
