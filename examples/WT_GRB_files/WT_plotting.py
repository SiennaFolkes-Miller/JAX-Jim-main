from anesthetic import read_chains
from anesthetic.plot import make_2d_axes

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

import pandas as pd
import numpy as np
from ast import literal_eval
import os


output_dir = "examples/WT_GRB_files"

analysis = "BNS"
validation = False         
toggle = 2                 
plot_truth = True    

if validation:
    samplesU_path = os.path.join(
        output_dir,
        "samples_BNS_unconditioned_real_main.csv"
    )
    samplesC_path = os.path.join(
        output_dir,
        "samples_BNS_conditioned_real_main.csv"
    )
    metadata_path = os.path.join(
        output_dir,
        "GW170817_data_main.csv"
    )
else:
    samplesU_path = os.path.join(
        output_dir,
        f"samples_BNS_unconditioned_{toggle}_main.csv"
    )
    samplesC_path = os.path.join(
        output_dir,
        f"samples_BNS_conditioned_{toggle}_main.csv"
    )
    metadata_path = os.path.join(
        output_dir,
        f"BNS_{toggle}_data_main.csv"
    )

loaded_samplesU = read_chains(samplesU_path)
loaded_samplesC = read_chains(samplesC_path)

metadata = pd.read_csv(metadata_path)

truths = None

if plot_truth and not validation:

    truths = {}

    for col in metadata.columns:
        if col.startswith("true_parameters."):

            name = col.split(".", 1)[1]

            truths[name] = metadata.loc[0, col]



plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    #'font.serif': ['Palatino Linotype', 'Book Antiqua', 'Georgia'],
    'font.size': 15,
})

params = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec", "lambda_1", "lambda_2"]
labels = [
    r"\mathcal{M}_{\mathrm{c}}", r"\mathrm{q}", r"\mathrm{s}_{1z}", r"\mathrm{s}_{2z}", r"\iota",
    r"d_{L}", r"t_{c}", r"\phi_{c}", r"\psi", r"\alpha", r"\delta", r"\Lambda_1", r"\Lambda_2"]

fig, axes = make_2d_axes(params, lower=True, diagonal=True, upper=True, figsize=(19.0, 13.0))

# STEP 1: Plot prior (pale for background)
prior_kws = dict(
    kinds=dict(lower='kde_2d', upper='scatter_2d', diagonal='kde_1d'),
    lower_kwargs=dict(color='#3B6E99', alpha=0.7),
    upper_kwargs=dict(color='#B8860B', alpha=0.7, lw=1.2),
    diagonal_kwargs=dict(color='gray', lw=1.5)
)
loaded_samplesU.plot_2d(axes, **prior_kws)

# Plot POSTERIOR (all three kinds: scatter, kde, kde)
posterior_kws = dict(
    kinds=dict(lower='kde_2d', upper='scatter_2d', diagonal='kde_1d'),
    lower_kwargs=dict(color='indigo', alpha=0.9),
    upper_kwargs=dict(color='crimson', alpha=0.9, markersize=1.4),
    diagonal_kwargs=dict(color='lightcoral', lw=1.7)
)
loaded_samplesC.plot_2d(axes, **posterior_kws)

# STEP 3: Axis labels
for i, param_x in enumerate(params):
    for j, param_y in enumerate(params):
        ax = axes[param_y][param_x]
        ax.set_xlabel(f"${labels[j]}$", fontsize=17)
        ax.set_ylabel(f"${labels[i]}$", fontsize=17)

# STEP 4: Truth lines
if truths is not None:
    axes.axlines(
        truths,
        color="black",
        lw=0.8,
        ls="--",
    )
# if truths is not None:
#     axes.scatter(
#         truths,
#         marker="o",
#         color="black",
#         s=16,
#         zorder=150,
#     )


# STEP 5: Top summary labels
for i, param in enumerate(params):
    x0 = 0.106
    spacing = 0.08335
    x_center = i * spacing + x0
    median = loaded_samplesU[param].quantile(0.5)
    low = loaded_samplesU[param].quantile(0.16)
    high = loaded_samplesU[param].quantile(0.84)
    lower = median - low
    upper = high - median

    label_text = (
        rf"${labels[i]} = {median:.0f}_{{-{lower:.0f}}}^{{+{upper:.0f}}}$"
        if param == "d_L" else
        rf"${labels[i]} = {median:.3g}_{{-{lower:.2g}}}^{{+{upper:.2g}}}$"
    )
    fig.text(x_center, 0.93, label_text, ha='center', va='bottom', fontsize=14)

# STEP 6: Tick/grid formatting
import matplotlib.ticker as ticker
for ax in axes.values.flatten():
    ax.tick_params(axis='both', labelsize=10.5)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# STEP 7: Title + Save
fig.suptitle("GW170817: Unconditioned vs Conditioned Posteriors", fontsize=21.5, y=0.995, fontweight='bold')
fig.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(
    os.path.join(output_dir, "BNS_Simulations.jpg"),
    dpi=600,
    bbox_inches="tight",
)
plt.show()