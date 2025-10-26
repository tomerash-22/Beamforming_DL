
# generting the dataset
mvdr_test is the main loop , contains in order : 
init_params : initalize all the structs 
config : configure all the matalb.phasedarray objects

loop: 
Randomizesim_param :generate the random parameters that change each iteration
sig_gen_ofdm : genrate an ofdm signal and the reciver imparied \ clean signal
MVDR  : calculate MVDR wieghts and snr gain
add_impair : create the structs for the learining prosses in python

# testing
both test_doa_after_coupling and test_snr_gain 
recives the input and outputs matrices and calculate the mentioned FOM's
