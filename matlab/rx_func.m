function [rx_sim_sig] = rx_func(cur_sig , cursig_param  ,ULA_0AZ_24_09,w_vec )

f_st = 0.5*sin( deg2rad(cursig_param.dir));
a_sv = [1 exp(1j*2*pi*f_st) exp(1j*4*pi*f_st) exp(1j*6*pi*f_st) ]';
% if (cursig_param.dir>0)
%     a_sv =flipud(a_sv);
% end
rx_sim_sig = (a_sv * cur_sig)' ; 


% 
% sp = ULA_0AZ_24_09.ElementSpacing;
% ang_calc =abs(cursig_param.dir);
% prog_velo=3e8;
% 
% distance_diff = sin(deg2rad(ang_calc))*sp;
% ele_amu=ULA_0AZ_24_09.NumElements;
% rx_sim_sig = zeros(ele_amu,length(cur_sig));
% time_diff = distance_diff/prog_velo;
% 
% if cursig_param.dir>0
%     rx_sim_sig(4,:) = cur_sig;
%     up=false;
%     ele_idx=ele_amu-1;
% else
%     rx_sim_sig(1,:) = cur_sig;
%     up=true;
%     ele_idx=2;
% end
% ii=0;
% % phase shift
% while(ii<ele_amu-1)     
% phase_mv = zeros(1,length(cur_sig));
% samp_diff = round(ele_idx*time_diff*cursig_param.fs);
% phase_mv(samp_diff+1)=1;
% rx_sim_sig(ele_idx,:)=cconv(cur_sig,phase_mv,length(cur_sig));
% rx_sim_sig(ele_idx,1:samp_diff+1) = ...
% awgn(zeros(size(1:samp_diff+1)) , cursig_param.snr);
% if up
%     ele_idx=ele_idx+1;
% else
%     ele_idx=ele_idx-1;
% end
% ii=ii+1;
% end
% % figure(3)
% % plot(linspace(-cursig_param.fs/2,cursig_param.fs/2,length(cur_sig)), ...
% %         db(abs(fftshift(fft((rx_sim_sig(1,:))))), 200));
% 
% 
% %gain from curr wieghts
% dbi_ang=directivity(ULA_0AZ_24_09,cursig_param.fc,cursig_param.dir,...
% 'Weights',w_vec);
% rx_sim_sig = rx_sim_sig.*db2mag(dbi_ang);
% % 
% figure(2)
% pattern(ULA_0AZ_24_09,cursig_param.fc,-180:180,0, ...
%     'Weights',w_vec,'Type','directivity',...
%     'PropagationSpeed',physconst('LightSpeed'),...
%     'CoordinateSystem','rectangular'); 
% 
% 
% 
% 
% 

