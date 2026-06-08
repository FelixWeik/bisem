import numpy as np
import pandas as pd
from tensorflow import keras
import phyloRNN as pn
import os
np.set_printoptions(suppress=True, precision=3)
from datetime import datetime
day_tag = datetime.now().strftime('%Y%m%d')
import time

t1=time.time()

sim = pn.simulator(n_taxa = 50,
                   n_sites = 1000,
                   n_eigen_features = 3,
                   min_rate = 0,  #
                   freq_uncorrelated_sites = 0.5,
                   freq_mixed_models = 0.40,
                   store_mixed_model_info = True,
                   tree_builder = 'nj',
                   subs_model_per_block = False,
                   phyml_path = None, #os.path.join(os.getcwd(), "phyloRNN"),
                   seqgen_path = None, # os.path.join(os.getcwd(), "phyloRNN", "seq-gen")
                   ali_path = os.path.join(os.getcwd(), "phyloRNN", "ali_tmp"),
                   DEBUG=False,
                   verbose = False,
                   min_avg_br_length=0.0002, # mean ~ 0.0025
                   max_avg_br_length=0.2
                   )


if __name__ == '__main__':
    # training set
    sim.reset_prms(CPUs = 50,
                   n_sims = 200,
                   data_name = "t50_s1000_mixed20",
                   base_seed = 1234)
    sim_file = pn.simulate_parallel(sim, return_outname=True)
    print("\nfinished at:", time.ctime(), "\nelapsed time:", round(time.time()-t1,2), "\n")

    
if False:
    #wd = os.path.dirname(sim_file)

    # test set
    sim.reset_prms(CPUs = 2,
                   n_sims = 300,
                   data_name = "test_clownfish_data",
                   run_phyml = True,
                   base_seed = 4321)
    #pn.simulate_parallel(sim)

    # train model
    # load data
    sim_file = "/home/silvestr/Documents/phyloRNN/clownfish_training_data20221008_e3_nj.npz"
    wd = os.path.dirname(sim_file)
    sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(sim_file,
                                                                         log_rates=False,
                                                                         log_tree_len=False,
                                                                         output_list=['per_site_rate', 'tree_len'],
                                                                         include_tree_features=False)

    n_onehot_classes = 4  # a, c, g, t
    (n_instances, n_sites, n) = sim['features_ali'].shape
    n_taxa = int(n / n_onehot_classes)
    tree_len_rescaler = 0.4 * n_taxa 
    dict_outputs["tree_len"] = dict_outputs["tree_len"] / tree_len_rescaler
    # dict_outputs["tree_len"] =  np.sqrt(dict_outputs["tree_len"])

    # build model
    node_list = [128,  # 0. sequence_LSTM_1
                 64,  # 1. sequence_LSTM_2
                 0,  # 2. phylo_FC_1 (dense on phylo features)
                 64,  # 3. site_NN and site_NN_tl (block-NNs dense with shared prms)
                 32,  # 4. site_rate_hidden
                 0,  # 5. sub_model_hidden
                 8,  # 6. tree_len_hidden
                 0  # 7. tree_len_hidden_2 (if > 0)
                 ]
    model_name = day_tag + "clownfish04"

    model = pn.build_rnn_model(n_sites=n_sites,
                               n_species=n_taxa,
                               n_eigenvec=0,
                               bidirectional_lstm=True,
                               loss_weights=[1, 1, 1],
                               nodes=node_list,
                               pool_per_site=True,
                               output_list=['per_site_rate', 'tree_len'],
                               mean_normalize_rates=True,
                               layers_norm=False,
                               separate_block_nn=True,
                               output_f=['softplus', 'softmax', 'softplus'],
                               optimizer=keras.optimizers.RMSprop(1e-3))

    # model.summary()
    print("N. model parameters:", model.count_params())

    # training
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=20,
                                               restore_best_weights=True)

    history = model.fit(dict_inputs, dict_outputs,
                        epochs=1000,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[early_stop],
                        batch_size=100)

    # save model
    pn.save_rnn_model(wd=wd,
                      history=history,
                      model=model, feature_rescaler=None, filename=model_name)

