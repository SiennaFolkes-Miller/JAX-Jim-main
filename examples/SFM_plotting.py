#taken exact imports from blackjax.py file

import anesthetic
from anesthetic import NestedSamples, MCMCSamples
import blackjax
from blackjax.smc.ess import ess as smc_ess
from blackjax.smc.resampling import systematic
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
#from tueplots import bundles
from tqdm import tqdm


# #HDF5
# from anesthetic import read_hdf

# samples = read_hdf(
#     "outdir/GW150914/nss_samples.h5",
#     "samples",
# )

# print(type(samples))
# print(samples.head())
# print(samples.columns)
# print(samples.shape)

#CSV
from pandas import read_csv
from anesthetic import NestedSamples

df = read_csv(
    "outdir/GW170817/nss_samples.csv",   #change between events
    skiprows=[1, 2],      #skip labels and weights rows
    index_col=0
)

nss_samples = NestedSamples(df)

plot_params = [
    "M_c",
    "q",
    "s1_z",
    "s2_z",
    "iota",
    "d_L",
    "t_c",
    "psi",
    "ra",
    "dec",
]

labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_z": r"$s_{1,z}$",
    "s2_z": r"$s_{2,z}$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
}


#making plot
fig, ax = anesthetic.make_2d_axes(plot_params, upper=False, figsize=(11, 9))
nss_samples.plot_2d(
    ax,
    kinds=dict(diagonal="kde_1d", lower="kde_2d"),
    label="NSS",
    color="C1",
)

#layout choices
for parameter, label in labels.items():
    nss_samples.set_label(parameter, label)
fig.tight_layout()

#saving image
fig.savefig("outdir/GW170817/NSS_corner.png")  #change between events
print("Saved corner plot to outdir/GW170817/NSS_corner.png")   #change between events