import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
np.set_printoptions(suppress=True, precision=3)
import time 
import phyloRNN as pn
from scipy.stats import gamma
import seaborn as sns

t1 = time.time()

model_wd = "path_to_trained_model"
model_name = "t28_s1000"
data_wd = "path_to_clownfish_data"
ali_file = "Chr1.WGAlign.FromBam.Filtered.fasta"
annotation_file = "Chr1.ExonAnnotation.Filtered.txt"

model_output_size = 1000
n_taxa = 28


# parse alignment file
ali_input = pn.parse_large_alignment_file(os.path.join(data_wd, ali_file),
                                          batch_size=model_output_size, 
                                          n_taxa=n_taxa) 
                                          # truncate=10000) # subset data for quicker run


np.savez_compressed(
    file =os.path.join(data_wd, ali_file + "_feat.npz"),
    features_ali=ali_input['sequence_data'])

# load model
trained_model = pn.load_rnn_model(wd=model_wd, filename="%s" % model_name)

# predict
emp_predictions = trained_model.predict(ali_input)
predicted_rel_rate = emp_predictions[0] # relative rates
predicted_tl = 10 ** emp_predictions[1] # tree length (back-transformed from a log10 prediction)
rates = predicted_rel_rate * predicted_tl # absolute rates

# save predictions
np.savez_compressed(
    file = os.path.join(data_wd, "%s_predictions.npz" % model_name),
    features_ali=ali_input['sequence_data'],
    rel_rates = predicted_rel_rate,
    tl = predicted_tl)

np.savetxt(os.path.join(data_wd, "%s_rates.txt" % model_name), rates)

