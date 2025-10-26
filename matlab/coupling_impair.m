function add_impair(rx_sig,impairments_param,dataset_param,impair) 
switch impair

    case 'coupling'
        R_sig = cov(rx_sig);
        R_coupled = R_sig*impairments_param.Z_mat;
        dataset_param.Z_mat(:,:,ii)=impairments_param.Z_mat;
        dataset_param.R_sig(:,:,ii) = R_sig;
        dataset_param.R_coupled(:,:,ii) = R_coupled;
    case 'IQ'

end
