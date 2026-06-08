import matplotlib.pyplot as plt
import numpy as np
import sys, os
import phyloRNN as pn
np.set_printoptions(suppress=True, precision=3)
import pandas as pd

path_phyml = os.path.join(os.getcwd(), "phyloRNN/", "bin")
wd = "/Users/dsilvestro/Software/phyloRNN/"
compare_file = os.path.join(wd, "testset_phylo/test_data20230306_e3_nj.npz")
model_wd = os.path.join(wd, "results/trained_models")
model_name = "20230504sepBLOCKresH128-8logTL"
summary_file = os.path.join(wd, "testset_phylo/LnL_model.tsv")
simulation_wd = "/Users/dsilvestro/Software/phyloRNN/testset_phylo/results20230306"
out_wd = "/Users/dsilvestro/Documents/Projects/Ongoing/phyloRNN/R1"

tbl = pd.read_csv(summary_file, sep="\t")
n_sites = 1000

######
n_sites = 1000
n_taxa = 50
(m, comp_predictions, comp_sim, comp_dict_inputs, comp_dict_outputs, all_res
 ) = pn.compare_rnn_phyml(compare_file,
                          model_file=model_name,
                          model_wd=model_wd,
                          output_dir=wd,
                          parse_heterogeneity_models=True,
                          log_tl=True,
                          plot_results=False)


######
rnn_predicted_rates = comp_predictions[0]
rnn_predicted_tl = 10**comp_predictions[1]
true_rates = comp_dict_outputs['per_site_rate']
true_tl = 10**comp_dict_outputs['tree_len']

ml_g_predicted_rates = []
ml_fr_predicted_rates = []
ml_g_predicted_tl = []
ml_fr_predicted_tl = []


for i in range(tbl.shape[0]): #
    ali_file_name = tbl.ali[i]
    sub_model = tbl.phyml_model[i]
    model_ind = np.argmax(np.array(['JC69', 'HKY85', 'GTR']) == sub_model)
    true_tree_file_name = ali_file_name + '.tre'

    # likelihood under the true tree
    p, l, lk_true = pn.run_phyml(os.path.join(simulation_wd, ali_file_name),
                        path_phyml,
                        model_ind,
                        n_sites,
                        ncat=4,
                        output_file=None,
                        tree_constraint=os.path.join(simulation_wd, true_tree_file_name),
                        topology_constraint=os.path.join(simulation_wd, true_tree_file_name),
                        remove_branch_length=True,
                        return_likelihoods=True
                        )

    ml_g_predicted_tl.append(float(l['tl_g']))
    ml_fr_predicted_tl.append(float(l['tl_fr']))

    ml_g_predicted_rates.append(np.array(p['rate_g']).flatten().astype(float))
    ml_fr_predicted_rates.append(np.array(p['rate_fr']).flatten().astype(float))

hm = []
for i in range(len(comp_sim['info'])):
    try:
        hm_str = comp_sim['info'][i]['rate_het_model'][0] + "_" + comp_sim['info'][i]['rate_het_model'][2]
    except:
        hm_str = comp_sim['info'][i]['rate_het_model'][-1] + "_" + comp_sim['info'][i]['rate_het_model'][2]
    hm.append(hm_str)
hm = np.array(hm)

res = {
    "heterogeneity_model": hm,
    "true_rates": true_rates,
    "rnn_predicted_rates": rnn_predicted_rates,
    "ml_g_predicted_rates": np.array(ml_g_predicted_rates),
    "ml_fr_predicted_rates": np.array(ml_fr_predicted_rates),
    "true_tl": true_tl,
    "rnn_predicted_tl": rnn_predicted_tl.flatten(),
    "ml_g_predicted_tl": np.array(ml_g_predicted_tl),
    "ml_fr_predicted_tl": np.array(ml_fr_predicted_tl)
}

pn.save_pkl(res, os.path.join(out_wd, "ml_est_true_tree_topology_only.pkl"))


######################################################
# re-load pickle file
######################################################
res = pn.load_pkl(os.path.join(out_wd, "ml_est_true_tree_topology_only.pkl"))

# calc MSE site-rates
print("MSE phyloRNN estimates:", np.mean((res['rnn_predicted_rates'] - res['true_rates'])**2))
print("MSE Gamma estimates:", np.mean(((res['ml_g_predicted_rates']) - res['true_rates'])**2))
print("MSE Free rates estimates:", np.mean(((res['ml_fr_predicted_rates']) - res['true_rates'])**2))

# calc MAPE tree length
print("MAPE phyloRNN estimates:", np.mean(np.abs(res['rnn_predicted_tl'] - res['true_tl']) / res['true_tl']))
print("MAPE Gamma estimates:", np.mean(np.abs(res['ml_g_predicted_tl'] - res['true_tl']) / res['true_tl']))
print("MAPE Free rates estimates:", np.mean(np.abs(res['ml_fr_predicted_tl'] - res['true_tl']) / res['true_tl']))

plt.hist((res['rnn_predicted_tl'] - res['true_tl']) / res['true_tl'])
plt.show()


hm = res['heterogeneity_model']
het_names = np.unique(hm)
het_indx_list = [np.where(hm == i)[0] for i in het_names]

# run for each heterogeneity model
all_res = {}
all_res['rates-all'] = pn.print_msa_compare(res['ml_fr_predicted_rates'],
                                            res['ml_g_predicted_rates'],
                                            res['rnn_predicted_rates'],
                                            res['true_rates'])

all_res['tl-all'] = pn.print_mape_compare(res['ml_fr_predicted_tl'],
                                          res['ml_g_predicted_tl'],
                                          res['rnn_predicted_tl'],
                                          res['true_tl'])

for i in range(len(het_names)):
    print("\n%s" % het_names[i])
    r = pn.print_msa_compare(res['ml_fr_predicted_rates'],
                             res['ml_g_predicted_rates'],
                             res['rnn_predicted_rates'],
                             res['true_rates'], het_indx_list[i])
    all_res['rates-' + het_names[i]] = r

for i in range(len(het_names)):
    print("\n Tree Length:", het_names[i])
    t = pn.print_mape_compare(res['ml_fr_predicted_tl'],
                          res['ml_g_predicted_tl'],
                          res['rnn_predicted_tl'],
                          res['true_tl'], het_indx_list[i])

    all_res['tl-' + het_names[i]] = t


# summarize results
res_tbl = []
for k in np.sort(list(all_res.keys())):
    r = all_res[k]
    tmp = [k]
    for i in range(3):
        try:
            tmp = tmp + [r['mse'][i], r['R2'][i]]
        except:
            tmp = tmp + [r['mape'][i], r['R2'][i]]
    res_tbl.append(tmp)

res_df = pd.DataFrame(res_tbl)
res_df.columns= ['Model',
                 'G-mse','G-R2',
                 'FR-mse', 'FR-R2',
                 'DL-mse', 'DL-R2']

res_df.to_csv(os.path.join(out_wd, "testset_results_logTL_true_tree_fix_topology.txt"),
              sep='\t',index=False)
