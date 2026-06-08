import os
import numpy as np
import phyloRNN as pn
from matplotlib import pyplot as plt
wd = "path_to_trained_model"
data_wd = "path_to_testset_data"
model_name = "t50_s1000"

simulate_alignments = False
plot = False
log_rates = False
start_sim = 0
n_sim = 100

# load trained phyloRNN model
trained_model = pn.load_rnn_model(os.path.join(wd, model_name))

# simulate data
sim = pn.simulator(n_taxa = 50,
                   n_sites = 1000,
                   n_eigen_features = 3,
                   min_rate = 0,  #
                   freq_uncorrelated_sites = 0.5,
                   freq_mixed_models = 0,
                   store_mixed_model_info = True,
                   tree_builder = 'nj',  
                   phyml_path = None,
                   seqgen_path = None, 
                   ali_path = data_wd, 
                   DEBUG=False,
                   verbose = True,
                   ali_schema = "nexus", 
                   min_avg_br_length=0.01,
                   max_avg_br_length=0.2
                   )

# run simulations set
for sim_i in range(start_sim, start_sim + n_sim):
    ali_name = os.path.join(data_wd, "ali%s" % sim_i)

    if simulate_alignments:
        res = sim.run_sim([sim_i, # rnd seed
                           1, # run one simulation
                           ali_name, # save alignment file
                           False] # run phyML
                           )

        ali_file = res[-1][0]['ali_file']

        # get features for rnn predictions
        true_site_rates = res[2][0]
        sim_res = {'features_ali': res[0][0],
                   'labels_rates': true_site_rates,
                   'labels_smodel': None,
                   'labels_tl': res[-2][0]
                   }

        # create input
        (comp_sim, dict_inputs, comp_dict_outputs
         ) = pn.rnn_in_out_dictionaries_from_sim(sim=sim_res,
                                                 log_rates=log_rates,
                                                 output_list=['per_site_rate', 'tree_len'],
                                                 include_tree_features=False)

        model_input = {'sequence_data': dict_inputs['sequence_data'].reshape((1,
                                                                              dict_inputs['sequence_data'].shape[0],
                                                                              dict_inputs['sequence_data'].shape[1]))
                       }

    else:
        ali_file = ali_name + ".nex"
        dat, model_input, dict_outputs = pn.parse_alignment_file(ali_file,
                                                               schema="nexus",
                                                               run_phyml_estimation=False,
                                                               save_nogap_ali=False
                                                               )



    print("Running predictions...")
    predictions = trained_model.predict(model_input)
    site_rates = predictions[0][0]

    # RevBayes script with Gamma rates
    pn.get_revBayes_script(ali_file,
                           sr=None, gamma_model=True, inv_model=True,
                           prior_bl=16.)
    
    # RevBayes script with phyloRNN rates discretized in 4 rate classes
    pn.get_revBayes_script(ali_file,
                           sr=site_rates, prior_bl=16.,
                           discretize_site_rate=4)

    # RevBayes script with phyloRNN rates discretized in 50 rate classes
    pn.get_revBayes_script(ali_file,
                           sr=site_rates, prior_bl=16.,
                           discretize_site_rate=50)

