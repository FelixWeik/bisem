## phyloRNN scripts

### Simulations
Scripts to generate training and test datasets with 50 taxa and alignments of 1000 nucleotides. 

`1.simulate_training_test_data.py` Simulate 60,000 training datasets with 50 tips and 1000 sites and 600 test datasets. 

`2.train_model.py` Train a `phyloRNN` model.

`3.compare_phyloRNN_w_phyML_1.py` Compare site rates and tree lengths  estimated on the test set based on a `phyloRNN` trained model and on `phyML` optimization. The output is a tab-separated table with accuracy metrics calculated for each model of rate heterogeneity. 

`4.compare_phyloRNN_w_phyML_2.py` Compare site rates and tree lengths  estimated on the test set based on a `phyloRNN` trained model and on `phyML` optimization using a fixed tree topology, constrained to the true one. The output is a tab-separated table with accuracy metrics calculated for each model of rate heterogeneity. 

`5.simulate_trainingset_mixed.py` Simulate training datasets with a larger fraction of alignments based on a mixed model of rate heterogeneity, to assess the potential improvement in the prediction accuracy for this subset of simulations.

---
### Clownfish_scripts
`1.simulate_and_train.py` Generate training set for the analysis of an alignment spanning chromosome 1 of 28 species of clownfish in batches of 1000 sites, and train a `phyloRNN` model.

`2.clownfish_predictions.py` Predict site rates across chromosome 1 of the clownfish clade (data available [here](https://datadryad.org/stash/share/rTkpzhuzwFtg9SxT4A4xT4oBLG2nxBf88fpqqmQkbvQ)). 

`3.plot_results.py` Parse exon annotation and plot results. 

---
### RevBayes_experiments
Scripts to generate training and test datasets and analyze them using `RevBayes`. Scripts 1-5 generate datasets with 100 sites, train a `phyloRNN` model and analyze them in `RevBayes` applying per-site rates and gamma distributed rates to test their effect on the accuracy of the estimated trees. 

Scripts 6-7 use previously a trained `phyloRNN` model with 20 taxa and 1000 sites, simulate a new test set, and analyze them in `RevBayes` applying per-site rates discretized into 4 or more rate categories. 

`1.simulate_train_data_revb.py` Simulate training and test data with 20 taxa and 100 sites. 

`2.train_model_revb.py` Train a new model based on training set (pre-trained model available in the **trained_models** directory). 

`3.generate_RevBayes_scripts.py` Predict per-site rates for the test set based on the trained model and create `RevBayes` scripts to run phylogenetic estimation using `phyloRNN` rates.

`4.parallelize_revb.py` Run `RevBayes` analyses in parallel. Note the the path to the alignment files is included in the generated `RevBayes` scripts, so their names and location must not be changed.

`5.1.parse_revb_res.py` and `5.2.parse_revb_res.R` Parse results of the `RevBayes` phylogenetic inference to assess the accuracy (weighted R-F distances) and compare models with gamma-distributed and `phyloRNN` rates. 

`6.sim_RevBayes_scripts_blocks.py` Simulate test data sets with 50 taxa and 1000 sites, predict site rates with pre-trained `phyloRNN` model (available in the **trained_models** directory), discretize them into 4 or more categories, create `RevBayes` scripts using estimated rate categories. 

`7.parse_revb_res_blocks.R` Parse results of the `RevBayes` phylogenetic inference to assess the accuracy (weighted R-F distances) and compare models with gamma-distributed and `phyloRNN` rate categories.

---
### Trained_models
Trained `phyloRNN` models used in our study. 

`t20_s100`: 20 tips, 100 sites (used for `RevBayes` tests)

`t28_s1000`: 28 tips, 1000 sites (used for clownfish dataset) 

`t50_s1000`: 50 tips, 1000 sites (used for general benchmarking and for `RevBayes` tests with discrete `phyloRNN` rate classes) 

`t50_s1000mixed20`: 50 tips, 1000 sites trained with higher fraction of mixed rate-model datasets (use to test accuracy on mixed models)
