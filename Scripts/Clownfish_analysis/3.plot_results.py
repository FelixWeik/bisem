import os
import sys
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pandas as pd
import os
np.set_printoptions(suppress=True, precision=3)
import time 
import phyloRNN as pn
from scipy.stats import gamma
import seaborn as sns
import pingouin as pg


data_wd = "path_to_clownfish_data"
ali_file = "Chr1.WGAlign.FromBam.Filtered.fasta"
annotation_file = "Chr1.ExonAnnotation.Filtered.txt"

model_output_size = 1000
n_taxa = 28

res = dict(np.load(os.path.join(data_wd, "t28_s1000_predictions.npz")))

predicted_rel_rate = res['rel_rates']
predicted_tl = res['tl']
rates = predicted_rel_rate * predicted_tl # absolute rates



### Stats
concat_predictions = rates.flatten()
print(np.mean(concat_predictions))
print(np.quantile(rates.flatten(), q=0.025), np.quantile(rates.flatten(), q=0.975))
print(np.mean(predicted_rel_rate)) # relative rates


# PLOT RATE HISTOGRAMS

# parse exon annotation
exon_tbl = pd.read_csv(os.path.join(data_wd, annotation_file), sep="\t")
exon_tbl_np = exon_tbl.to_numpy()

exon_len = exon_tbl_np[:,2] - exon_tbl_np[:,1]

exon_selected = exon_tbl_np[np.where(exon_len > 500)[0],:]
rs = pn.get_rnd_gen(4321)
exon_selected = exon_selected[rs.choice(
    range(exon_selected.shape[0]),
    exon_selected.shape[0],
    replace=False)]


def renormalize(x):
    # turn rates into relative rates, i.e. mean = 1, as in a gamma model
    return x / np.mean(x)


fig, axes = plt.subplots(3, 3, figsize=(10, 7))
truncate_plot = 100
x_min, x_max = 0, 3
exon_n = 0
counter = 0
for ax in axes:
    if counter == 0:
        x = renormalize(concat_predictions[concat_predictions < truncate_plot])
        _ = ax[0].hist(x, bins=500, density=True)
        fit_alpha, fit_loc, fit_beta = gamma.fit(x[rs.choice(range(len(x)), 100000)], floc=False)
        x_ax= np.linspace(x_min, x_max, 1000)
        ax[0].plot(x_ax, gamma.pdf(x_ax, fit_alpha, fit_loc, fit_beta),
                'r-', lw=2, alpha=0.6, label='gamma pdf')
        ax[0].set_xlim(xmin=x_min, xmax=x_max)
    
    else:
        indx_exone = exon_selected[exon_n][1:]  # 1
        x = renormalize(concat_predictions[indx_exone[0]:indx_exone[1]])
        _ = ax[0].hist(x, bins=50, density=True)
        fit_alpha, fit_loc, fit_beta = gamma.fit(x, floc=False)
        [x_min, x_max] = np.quantile(x, q=[0.01, 0.99])  * np.array([0.8, 1.2])
        x_ax = np.linspace(0.00001, 3, 1000)
        ax[0].plot(x_ax, gamma.pdf(x_ax, fit_alpha, fit_loc, fit_beta),
                   'r-', lw=2, alpha=0.6, label='gamma pdf')
        ax[0].set_xlim(xmin=x_min, xmax=x_max)
    exon_n += 1
    indx_exone = exon_selected[exon_n][1:]  # 2
    x = renormalize(concat_predictions[indx_exone[0]:indx_exone[1]])
    _ = ax[1].hist(x, bins=50, density=True)
    fit_alpha, fit_loc, fit_beta = gamma.fit(x, floc=False)
    [x_min, x_max] = np.quantile(x, q=[0.01, 0.99]) * np.array([0.8, 1.2])
    x_ax = np.linspace(0.00001, 3, 1000)
    ax[1].plot(x_ax, gamma.pdf(x_ax, fit_alpha, fit_loc, fit_beta),
               'r-', lw=2, alpha=0.6, label='gamma pdf')
    ax[1].set_xlim(xmin=x_min, xmax=x_max)
    exon_n += 1
    indx_exone = exon_selected[exon_n][1:]  # 3
    x = renormalize(concat_predictions[indx_exone[0]:indx_exone[1]])
    _ = ax[2].hist(x, bins=50, density=True)
    fit_alpha, fit_loc, fit_beta = gamma.fit(x, floc=False)
    [x_min, x_max] = np.quantile(x, q=[0.01, 0.99]) * np.array([0.8, 1.2])
    x_ax = np.linspace(0.00001, 3, 1000)
    ax[2].plot(x_ax, gamma.pdf(x_ax, fit_alpha, fit_loc, fit_beta),
               'r-', lw=2, alpha=0.6, label='gamma pdf')
    ax[2].set_xlim(xmin=x_min, xmax=x_max)
    exon_n += 1
    counter += 1

fig.tight_layout()
fig.show()
fig.savefig(os.path.join(data_wd, "clownfish_srate_histograms.pdf"))


# PLOT Tree length across 10M SITES
# number of substitutions per site across all taxa
n_substitutions_per_site = predicted_tl.flatten()

fig, axes = plt.subplots(10, 1, figsize=(90 * 0.2, 30 * 0.2))
# fig.tight_layout(pad=5)
sites_per_row = 1000
i=0
vmin = np.min(n_substitutions_per_site)
vmax = np.max(n_substitutions_per_site) * 0.75

for ax in axes:
    r = np.arange(i,i + sites_per_row)
    y = n_substitutions_per_site[r]
    
    sns.heatmap(
        y.reshape((1, sites_per_row)),
        cmap="viridis",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
    )
    i += sites_per_row

fig.tight_layout()
fig.show()
fig.savefig(os.path.join(data_wd, "clownfish_substitutions_per_site.pdf"))



# PLOT exons vs introns
def estMeanRate(start, stop, toprint=False):
    meanRate = list() #mean rate for the exon
    n = 0 #number of rates to average
    
    startRow = int((start - 1) / 1000) #no need to remove 1 as 0-999 on row 0, 1000-1999 on row 1, etc...
    startCol = ((start - 1) % 1000) #need to remove 1 as pos 1 is in col 0, 
    
    stopRow = int((stop - 1) / 1000)
    stopCol = ((stop - 1) % 1000)
    
    if toprint: print(f"---- \nIn estMeanRate:\nstart={start}, stop={stop}\n")
    #same row for the start and end -> simply loop through the columns
    if startRow == stopRow:
        if toprint: print(f"On the same row: {startRow}") 
        if toprint: print(f"\tgoing from {startCol} to {stopCol}") 
        for c in range(startCol, stopCol+1):
            meanRate.append(rates[startRow, c])
            n += 1
    else:
        if toprint: print(f"Not on the same row: {startRow} - {stopRow}") 
        for r in range(startRow, stopRow+1):
            if r == startRow: #first row, go from startCol to 1000
                if toprint: print(f"\tgoing from {startCol} to 999 on row {r}")
                for c in range(startCol, 1000): #1000 cols in the df
                    meanRate.append(rates[r, c])
                    n += 1
            elif r == stopRow: #last row, go from 0 to stopCol
                if toprint: print(f"\tgoing from 0 to {stopCol} on row {r}") 
                for c in range(0, stopCol+1):
                    meanRate.append(rates[r, c])
                    n += 1
            else: #row between the start and stop, take the 1000 values
                if toprint: print(f"\tgoing from 0 to 999 on row {r}")
                for c in range(0, 1000):
                    meanRate.append(rates[r, c])
                    n += 1
    if n == 0:
        print(f"Problem with window: {start} - {stop}. Skipping it")
        return(0)
    else:
        return([meanRate, n])


exons = pd.read_csv(os.path.join(data_wd, annotation_file), sep="\t")

f = open(os.path.join(data_wd, "exon_intron_rates.txt"), "w")
f.write("start,stop,before,exon,after\n")

limit = 250

for idx, line in exons.iterrows():
    #get rates for current exons
    start = line.ExonStart
    stop = line.ExonStop
    
    if idx == 0:
        stopBefore = 1
    else:
        stopBefore = exons.iloc[idx-1][2] + 1
    
    if idx == len(exons) - 1:
        startAfter = len(rates) * 1000
    else:
        startAfter = exons.iloc[idx+1][1] - 1
    
    if (stop - start) < limit or (start - stopBefore) < limit or (startAfter - stop) < limit:
        continue
    
    before = estMeanRate(start - 1 - limit, start - 1, False)
    exon = estMeanRate(start, stop, False)
    after = estMeanRate(stop + 1, stop + 1 + limit, False)
    
    meanRate = [sum(before[0]) / before[1], sum(exon[0]) / exon[1], sum(after[0]) / after[1]]
    f.write(f"{start},{stop},{str(meanRate[0])},{str(meanRate[1])},{str(meanRate[2])}\n")

f.close()

# run paired t-tests
exon_intr_rates = pd.read_csv(os.path.join(data_wd, "exon_intron_rates___.txt"))

ex = exon_intr_rates['exon'].to_numpy()
f1 = exon_intr_rates['before'].to_numpy()
f2 = exon_intr_rates['after'].to_numpy()

t1 = pg.ttest(ex, f1, paired=True)
print(t1.T)
t1 = pg.ttest(ex, f2, paired=True)
print(t1.T)
t1 = pg.ttest(f1, f2, paired=True)
print(t1.T)
print("Mean rate change", 100 * 1 - np.mean(ex / f1))


### Rate barplot
exon_intr_rates["non_exon"] = np.mean([exon_intr_rates["before"], exon_intr_rates["after"]], axis=0)
x = [exon_intr_rates["exon"], exon_intr_rates["non_exon"]]

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
boxplots = ax.boxplot(x, notch=True, sym="", vert=None, whis=True, widths=0.5, meanline=True)

ymin, ymax = ax.get_ylim() # 0.14, 0.47
xmin, xmax = ax.get_xlim()

Nx, Ny = 1, 10000
imgArr = np.tile(np.linspace(0, 1, Ny), (Nx, 1)).T
cmap = 'viridis'

for boxplot in boxplots['boxes']:
    path = Path(boxplot.get_path().vertices)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)
    img = ax.imshow(imgArr, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="auto", cmap=cmap, clip_path=patch)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Exonic region", "Non-exonic regions"])
ax.set_ylabel("")
ax.set_yticks([])
ax.set_yticklabels([])
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="5%", pad="2%")
norm = Normalize(vmin=ymin, vmax=ymax)
cb = ColorbarBase(cax, cmap=get_cmap(cmap), norm=norm, orientation='vertical')
cb.set_label("Substitutions per site")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(data_wd,"comparison_exons.pdf"))
plt.close("all")
plt.clf()

