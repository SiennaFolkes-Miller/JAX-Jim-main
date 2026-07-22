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
import scipy
from scipy.stats import gaussian_kde


##READ IN DATA ##
#CSV
from pandas import read_csv
from anesthetic import NestedSamples

# df = read_csv(
#     "outdir/GW170817/nss_samples.csv",   #change between events
#     skiprows=[1, 2],      #skip labels and weights rows
#     index_col=0
# )

# nss_samples = NestedSamples(df)

from anesthetic import read_chains

nss_samples = read_chains("outdir/GW170817/nss_samples.csv")

##CORNER PLOT
plot_params = [
    "M_c",
    "q",
    "s1_z",
    "s2_z",
    "lambda_1",
    "lambda_2",
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



##H0 POSTERIOR##
dL = np.asarray(nss_samples["d_L"])
weights = np.asarray(nss_samples.get_weights())


def weighted_quantile(values, weights, quantiles):

    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    return np.interp(quantiles, cumulative, values)



weighted_mean_dL = np.average(dL, weights=weights)
q16_dL, q50_dL, q84_dL = weighted_quantile(dL,weights,[0.16, 0.50, 0.84])

print(f"dL = {q50_dL:.1f} +{q84_dL-q50_dL:.1f} -{q50_dL-q16_dL:.1f} Mpc")


kde = gaussian_kde(dL, weights=weights)

x = np.linspace(dL.min(), dL.max(), 1000)
pdf = kde(x)

mode = x[np.argmax(pdf)]

print(f"MAP = {mode}, weighted mean = {weighted_mean_dL}")



#Using velocities from Ming et al, all velocities in km/s and dL in Mpc
vr = 3327
sigma_vr = 72
vp = 310
sigma_vp = 150


n_samples = len(dL)
vr_dist = scipy.stats.norm(vr, sigma_vr)
vp_dist = scipy.stats.norm(vp, sigma_vp)
vr_samples = vr_dist.rvs(n_samples)
vp_samples = vp_dist.rvs(n_samples)
H0 = (vr_samples - vp_samples)/dL


weighted_mean_H0 = np.average(H0, weights=weights)
q16_H0, q50_H0, q84_H0 = weighted_quantile(H0,weights,[0.16, 0.50, 0.84])

print(f"H0 = {q50_H0:.1f} +{q84_H0-q50_H0:.1f} -{q50_H0-q16_H0:.1f} km/s/Mpc")



kde = gaussian_kde(H0, weights=weights)

x = np.linspace(H0.min(), H0.max(), 1000)
pdf = kde(x)

mode = x[np.argmax(pdf)]

print(f"MAP = {mode}, weighted mean = {weighted_mean_H0}")



fig, axes = plt.subplots(1, 2,figsize=(12, 5))


axes[0].hist(dL, bins=100, weights=weights, density=True, alpha=0.6)
axes[0].axvline(q50_dL, color='black', lw=2, label='Median')
axes[0].axvline(q16_dL, color='gray', ls='--', lw=1, label='16% quartile')
axes[0].axvline(q84_dL, color='gray', ls='--', lw=1, label='84% quartile')
axes[0].set_xlabel(r"$d_L$ (Mpc)", fontsize=14)
axes[0].set_ylabel("Posterior probability density", fontsize=16)


axes[1].hist(H0, bins=100, weights=weights, density=True, alpha=0.6)
axes[1].axvline(q50_H0, color='black', lw=2, label='Median')
axes[1].axvline(q16_H0, color='gray', ls='--', lw=1, label='16% quartile')
axes[1].axvline(q84_H0, color='gray', ls='--', lw=1, label='84% quartile')
axes[1].set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)", fontsize=14)
axes[1].legend()
axes[1].set_xlim(0, 150)
fig.savefig("outdir/GW170817/H0_posterior.png")
print("Saved corner plot to outdir/GW170817/H0_posterior.png") 

plt.show()