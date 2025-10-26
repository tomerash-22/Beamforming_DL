function [snr_gain] = snr_calc(w,sig_est_dir,ULA_0AZ_24_09,sig_param, ...
    inter_param,source )


fc=sig_param.fc;
inputAngle = sig_param.inputAngle;

del_f = inter_param.del_f;
inter_ang = inter_param.inputAngle;
inter_att = inter_param.attindB;

pat = pattern(ULA_0AZ_24_09 ,sig_param.fc,-90:90,0,'Type','powerdb',...
      'PropagationSpeed',physconst('LightSpeed'),'Weights',...
      fliplr(w),'Normalize',false,...
      'CoordinateSystem','rectangular');

switch source
   
     case 'MATLAB'
        sig_gain_bf = pat(91 + sig_est_dir);
        inter_gain_bf = pat(91 + inter_ang(1,1));
        noise_gain_bf=(sum(pat)/180); %sum only non -inf beams
        inter_plus_noise_bf = mag2db(db2mag(inter_gain_bf - inter_att) + ...
            db2mag(noise_gain_bf));
        snr_gain =  (sig_gain_bf - inter_plus_noise_bf)/2 ; 
    otherwise
        sig_gain = pat(91 + sig_est_dir);
        inter_gain = pat(91 - inter_ang(1,1));
        noise_gain=(sum(pat)/180); %sum only non -inf beams
        inter_plus_noise = mag2db(db2mag(inter_gain - inter_att) + ...
        db2mag(noise_gain));
        snr_gain = (sig_gain - inter_plus_noise)/2 ; 
        
end
