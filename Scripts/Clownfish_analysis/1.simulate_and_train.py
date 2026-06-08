import numpy as np
import pandas as pd
from tensorflow import keras
import phyloRNN as pn
import os
np.set_printoptions(suppress=True, precision=3)
from datetime import datetime
day_tag = datetime.now().strftime('%Y%m%d')
import time

n_taxa = 28
n_sites = 1000

sim = pn.simulator(n_taxa = n_taxa,
                   n_sites = n_sites,
                   n_eigen_features = 3,
                   min_rate = 0,
                   freq_uncorrelated_sites = 0.5,
                   freq_mixed_models = 0.05,
                   store_mixed_model_info = True,
                   tree_builder = 'nj',
                   subs_model_per_block = False,
                   phyml_path = None,
                   seqgen_path = None,
                   ali_path = os.path.join(os.getcwd(), "phyloRNN", "ali_tmp"),
                   DEBUG=False,
                   verbose = False,
                   min_avg_br_length=0.0002, # mean ~ 0.0025
                   max_avg_br_length=0.01
                   )


if __name__ == '__main__':
    # training set (~10,000 datasets)
    t1=time.time()
    sim.reset_prms(CPUs = 64,
                   n_sims = 157,
                   data_name = "clownfish_training",
                   base_seed = 1234)
    training_file = pn.simulate_parallel(sim, return_outname=True)
    print("\nfinished at:", time.ctime(), "\nelapsed time:", round(time.time()-t1,2), "\n")
    
    # load training data
    t1=time.time()
    sim, dict_inputs, dict_outputs = pn.rnn_in_out_dictionaries_from_sim(training_file,
                                                                         log_rates=False,
                                                                         log_tree_len=True,
                                                                         output_list=['per_site_rate','tree_len'],
                                                                         include_tree_features=False)

    # setup model architecture
    model_config = pn.rnn_config(n_sites=n_sites, n_taxa=n_taxa) # default settings

    # build model
    model = pn.build_rnn_model(model_config,
                               optimizer=pn.keras.optimizers.RMSprop(1e-3),
                               print_summary=False)


    # training
    early_stop = pn.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=5,
                                                  restore_best_weights=True)

    history = model.fit(dict_inputs, dict_outputs,
                        epochs=1000,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[early_stop],
                        batch_size=100)

    # save model
    wd = os.path.dirname(training_file)
    model_name = "t28_s1000"
    pn.save_rnn_model(wd=wd, history=history, model=model, filename=model_name)

    print("\nfinished at:", time.ctime(), "\nelapsed time:", round(time.time()-t1,2), "\n")
    
