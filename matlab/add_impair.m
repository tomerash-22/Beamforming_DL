function [dataset_param]= add_impair(signal_data,impairments_param,dataset_param,ii,impair , sig_len) 

switch impair

    case 'coupling'
        R_sig = signal_data.R_sig{1} ; 
        R_coupled = R_sig*impairments_param.Z_mat;
        dataset_param.Z_mat(:,:,ii)=impairments_param.Z_mat;
        dataset_param.R_sig(:,:,ii) = R_sig;
        dataset_param.R_coupled(:,:,ii) = R_coupled;
    case 'IQ'
       % Append new signals to the fields
       rx_sig= signal_data.rx_sig_ofdm{1};
       rx_tar=signal_data.rx_tar{1};
       for ele = 1:1:4
        dataset_param.clean_sig{end + 1} =rx_tar(ele,1:sig_len)';
        dataset_param.distor_sig{end + 1} =rx_sig(ele,1:sig_len)';
       end
    case 'FCO'
        rx_tar=signal_data.rx_tar{1};
        for ele = 1:1:4
        dataset_param.clean_sig{end + 1} =rx_tar(ele,1:sig_len)';
        dataset_param.distor_sig{end + 1} =rx_tar(ele,1:sig_len)'.*exp(2*pi*1j*impairments_param.fco);
        dataset_param.fco{end+1} = impairments_param.fco;
        end
end

