function [sig_param , inter_param , ch_param,impairments_param] =...
Randomizesim_param(sig_param , inter_param , ch_param,impairments_param)
ch_param.snr = randi([-20,50]) ; 
inter_param.attindB = randi([-30,10]) ;

sig_param.dir = randi([-70,70]);
sig_param.inputAngle = [sig_param.dir;0];
catch_flag=false;

 
impairments_param.gain=randi([-15 15] , [1,4])/10; %in db
impairments_param.phase=(randi([-10 10],[1,4])); %in deg

sig_param.ofdm_BW = randi([18e6,22e6]);

impairments_param.fco = randn*10e3;



try
    pos_dir = randi([sig_param.dir+20,80]);
catch
    %warning('dir+10 not avlid')
    inter_param.dir = randi([-80,sig_param.dir-10]);
    catch_flag=true;
end

try
    neg_dir = randi([-80,sig_param.dir-10]);
catch
    %warning('dir-10 not avlid')
    inter_param.dir = pos_dir;
    catch_flag=true;
end

if (~catch_flag)
    if (randn(1)>0)
        inter_param.dir = pos_dir;
    else
        inter_param.dir = neg_dir;
    end
end
inter_param.inputAngle = [inter_param.dir;0];

Z_rand = diag(ones(1,4)) ;
Z_1diff =ones(1,6).* db2mag (randi([-50,-5]));
Z_2diff =ones(1,4).* db2mag (randi([-500,-50]));
Z_1diff =awgn(Z_1diff,-1*mag2db(Z_1diff(1,1))+50);
Z_2diff =awgn(Z_2diff,-1*mag2db(Z_2diff(1,1))+50);

ii=1;
for j=1:1:3
Z_rand(j,j+1) = Z_1diff(1,ii);
ii=ii+1;
Z_rand(j+1,j) = Z_1diff(1,ii);
ii=ii+1;
end

ii=1;
for j=1:1:2
Z_rand(j,j+2) = Z_2diff(1,ii);
ii=ii+1;
Z_rand(j+2,j) = Z_2diff(1,ii);
ii=ii+1;
end
impairments_param.Z_mat = Z_rand;

