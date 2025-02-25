function BF_Flag = BF_failed(sig_param,inter_param ,ULA_0AZ_24_09,w_mvdr,w_matlab,snr_gain )
%gain from curr wieghts
dbi_ang_mvdr = directivity(ULA_0AZ_24_09,sig_param.fc,-90:90,...
'Weights',w_mvdr);
dbi_ang_matlab = directivity(ULA_0AZ_24_09,sig_param.fc,-90:90,...
'Weights',w_matlab);

snr_gain.mvdr=snr_calc(w_mvdr,sig_param.dir,ULA_0AZ_24_09,sig_param ...
    ,inter_param,'NVDR');
snr_gain.matlab=snr_calc(w_matlab,sig_param.dir,ULA_0AZ_24_09,sig_param ...
    ,inter_param,'MATLAB');
snr_gain.mvdr_fl=snr_calc(flipud (w_mvdr),sig_param.dir,ULA_0AZ_24_09,sig_param ...
    ,inter_param,'MATLAB');
snr_gain.matlab_fl=snr_calc(flipud (w_matlab),sig_param.dir,ULA_0AZ_24_09,sig_param ...
    ,inter_param,'MATLAB');
snr_gain_arr = [snr_gain.mvdr , snr_gain.matlab , snr_gain.mvdr_fl ,snr_gain.matlab_fl  ];
% snr_gain.mvdr=calc(dbi_ang_mvdr,inter_param,sig_param);
% snr_gain.matlab=calc(dbi_ang_matlab,inter_param,sig_param);
% figure(2)
% pattern(ULA_0AZ_24_09,sig_param.fc,-90:90,0, ...
%     'Weights',w_mvdr,'Type','directivity',...
%     'PropagationSpeed',physconst('LightSpeed'),...
%     'CoordinateSystem','rectangular'); 

if(max(snr_gain_arr)<snr_gain.snrgain_th) 
    BF_Flag =true;
else
    BF_Flag =false;
end
end

% function [snrcalc]=calc (dbi_ang,inter_param,sig_param )
% 
% snr_calc()
% 
% % dbi_sig=dbi_ang(sig_param.dir+91);
% % dbi_inter=dbi_ang(inter_param.dir+91);
% % noise_lv = sum(dbi_ang)/180;
% % inter_and_noise_g = mag2db(db2mag(dbi_inter - inter_param.attindB) + ...
% %             db2mag(noise_lv));
% % snrcalc =  dbi_sig - inter_and_noise_g; 
% end
