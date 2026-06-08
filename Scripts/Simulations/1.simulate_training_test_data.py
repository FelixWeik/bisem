import os
import phyloRNN as pn


sim = pn.simulator(
                   n_taxa = 50,
                   n_sites = 1000,
                   n_eigen_features = 3,
                   min_rate = 0,  #
                   freq_uncorrelated_sites = 0.5,
                   freq_mixed_models = 0.05,
                   store_mixed_model_info = True,
                   tree_builder = 'nj',  
                   subs_model_per_block = False,  
                   phyml_path = os.path.join(os.getcwd(), "phyloRNN", "bin"),  # path to phyml and seq binaries
                   seqgen_path = os.path.join(os.getcwd(), "phyloRNN", "bin"), # if None, it will try to use
                   ali_path = None,                                            # system-wide installed software
                   DEBUG=False,
                   verbose = False,
                   )


if __name__ == '__main__':
    # training set
    sim.reset_prms(CPUs = 60,
                   n_sims = 1000,
                   data_name = "training_data",
                   base_seed = 1234)
    pn.simulate_parallel(sim)

    # test set
    sim.reset_prms(CPUs = 2,
                   n_sims = 300,
                   data_name = "test_data",
                   run_phyml = True, # if set to True, phyML is run on the simulated alignemnt
                   base_seed = 4321)
    pn.simulate_parallel(sim)
