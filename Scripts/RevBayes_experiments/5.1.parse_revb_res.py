import os
import numpy as np
import phyloRNN as pn
from matplotlib import pyplot as plt
from pylabeledrf.computeLRF import *
import dendropy
import pandas as pd


data_dir = "path_to_revbayes_output"
output_dir = "path_to_summary_file"
n_sim = 200
burnin = 800

delta_lik = []
res_list = []
for sim_i in range(n_sim):
    ali_name = os.path.join(data_dir, "ali%s" % sim_i)

    pkl_f = "%s_info.pkl" % ali_name
    log_G_f = "%s_G.log" % ali_name
    log_DL_f = "%s_DL.log" % ali_name

    sim_data = pn.load_pkl(pkl_f)

    # likelihood
    mean_lik_G = np.mean(np.loadtxt(log_G_f, skiprows=burnin)[:,2])
    mean_lik_DL = np.mean(np.loadtxt(log_DL_f, skiprows=burnin)[:,2])
    delta_lik.append(mean_lik_G - mean_lik_DL)

    # collect results
    res_list.append(["ali%s" % sim_i,
                     sim_data[-1][0]['rate_het_model'][0] + "-" + sim_data[-1][0]['rate_het_model'][-1],
                     tl_true,
                     mean_lik_G - mean_lik_DL
                     ])




# res table
res_df = pd.DataFrame(res_list)
res_df.columns = ["sim", "model", "tree_length", "dLogLik"]
res_df.to_csv(os.path.join(output_dir, "summary_res.txt"),
              sep='\t', index=False)

