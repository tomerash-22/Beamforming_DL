function [rx_sim_sig] = rx_func(cur_sig , cursig_param  ,ULA_0AZ_24_09,w_vec )

f_st = 0.5*sin( deg2rad(cursig_param.dir));
a_sv = [1 exp(1j*2*pi*f_st) exp(1j*4*pi*f_st) exp(1j*6*pi*f_st) ]';
rx_sim_sig = (a_sv * cur_sig)' ; 

