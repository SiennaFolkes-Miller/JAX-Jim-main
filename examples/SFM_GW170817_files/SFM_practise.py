
############################
##copied GW150914 pipeline##
############################






import argparse
import json
import os

import anesthetic
from anesthetic import NestedSamples, MCMCSamples
import blackjax
from blackjax.smc.ess import ess as smc_ess
from blackjax.smc.resampling import systematic
import jax
jax.config.update("jax_enable_x64", True) #(LC)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
#from tueplots import bundles #(LC)
from tqdm import tqdm

from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    SinePrior,
    UniformPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.likelihood import TransientLikelihoodFD #(LC)
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD #(LC)
#from jimgw.core.single_event.likelihood import HeterodynedPhaseMarginalizedLikelihoodFD #(LC)
from jimgw.core.single_event.transforms import (
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
)
from jimgw.core.single_event.transform_utils import eta_to_q   #(LC) different file name
#from jimgw.core.single_event.waveform import RippleIMRPhenomXAS (LC)
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2   #new waveform (LC)

#plt.rcParams.update(bundles.tmlr2023()) #(LC)


##VISUALISING INITIAL PARTICLES##
#note cosine prior [-pi/2, pi/2] and sine prior [0, pi]

def build_prior() -> CombinePrior:
    """Prior matching the GW150914 setup."""

    prior = []
    RA_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
    #Dec_prior = UniformPrior(-jnp.pi/2, jnp.pi/2, parameter_names=["dec"])
    #RA_prior = CosinePrior(parameter_names=["ra"])
    Dec_prior = CosinePrior(parameter_names=["dec"])   
    dL_prior = UniformPrior(10.0, 75.0, parameter_names=["d_L"])
    iota_prior = SinePrior(parameter_names=['iota'])

    prior.extend([RA_prior, Dec_prior, dL_prior, iota_prior])
    return CombinePrior(prior)


#sample prior
prior = build_prior()


rng_key = jax.random.PRNGKey(42)
num_live = 5000
num_delete = num_live//2


samples = prior.sample(rng_key, num_live)


ra = samples["ra"]
dec = samples["dec"]
dL = samples["d_L"]
iota = samples["iota"]
# convert to numpy for matplotlib
ra = jnp.asarray(ra) - jnp.pi
dec = jnp.asarray(dec)
dL = jnp.asarray(dL)
iota = jnp.asarray(iota)

fig = plt.figure(figsize=(12, 10))

# Top-left: Mollweide projection
ax1 = fig.add_subplot(221, projection="mollweide")

hist = ax1.hist2d(ra, dec, bins=50, cmap="viridis", alpha=0.5)
plt.colorbar(hist[3], ax=ax1, label="Number of particles")
ax1.scatter(ra, dec, s=1, color="k", alpha=0.2)

ax1.grid(True)
ax1.set_xlabel("RA")
ax1.set_ylabel("Dec")

# Top-right: RA vs iota
ax2 = fig.add_subplot(222)

hist = ax2.hist2d(ra, iota, bins=50, cmap="viridis")
plt.colorbar(hist[3], ax=ax2, label="Number of particles")
ax2.scatter(ra, iota, s=1, color="k", alpha=0.2)

ax2.set_xlabel("RA")
ax2.set_ylabel(r"$\iota$")
ax2.grid(True)

# Bottom-left: RA vs dL
ax3 = fig.add_subplot(223)

hist = ax3.hist2d(ra, dL, bins=50, cmap="viridis")
plt.colorbar(hist[3], ax=ax3, label="Number of particles")
ax3.scatter(ra, dL, s=1, color="k", alpha=0.2)

ax3.set_xlabel("RA")
ax3.set_ylabel(r"$d_L$")
ax3.grid(True)

# Bottom-right: dL vs iota
ax4 = fig.add_subplot(224)

hist = ax4.hist2d(dL, iota, bins=50, cmap="viridis")
plt.colorbar(hist[3], ax=ax4, label="Number of particles")
ax4.scatter(dL, iota, s=1, color="k", alpha=0.2)

ax4.set_xlabel(r"$d_L$")
ax4.set_ylabel(r"$\iota$")
ax4.grid(True)

plt.tight_layout()
plt.show()


##VISUALISATION DEAD POINTS AND FINAL LIVE POINTS##
#CSV
from pandas import read_csv
from anesthetic import NestedSamples

df = read_csv(
    "outdir/GW170817/nss_samples.csv",   #change between events
    skiprows=[1, 2],      #skip labels and weights rows
    index_col=0
)

nss_samples = NestedSamples(df)

ref_RA = 3.39736826 - jnp.pi
ref_Dec = -0.34000186
ref_dL = 40.06331818
ref_iota = 1.93095421

ra = jnp.asarray(nss_samples["ra"]) - jnp.pi
dec = jnp.asarray(nss_samples["dec"])
dL = jnp.asarray(nss_samples["d_L"])
iota = jnp.asarray(nss_samples["iota"])
from matplotlib.animation import FuncAnimation
import numpy as np

# Convert to NumPy (important for animation)
ra_np = np.asarray(ra)
dec_np = np.asarray(dec)
dL_np = np.asarray(dL)
iota_np = np.asarray(iota)

fig = plt.figure(figsize=(12, 10))

# ------------------ RA vs Dec ------------------
ax1 = fig.add_subplot(221, projection="mollweide")
scat1 = ax1.scatter([], [], s=2, color="m", alpha=0.3)
ax1.scatter(ref_RA, ref_Dec, marker="x", color="black", s=50)
ax1.set_title("RA vs Dec")
ax1.grid(True)

# ------------------ RA vs dL -------------------
ax2 = fig.add_subplot(222)
scat2 = ax2.scatter([], [], s=2, color="tab:blue", alpha=0.3)
ax2.scatter(ref_RA, ref_dL, marker="x", color="black", s=50)
ax2.set_title("RA vs dL")
ax2.set_xlabel("RA")
ax2.set_ylabel(r"$d_L$")
ax2.grid(True)

# ------------------ RA vs iota -----------------
ax3 = fig.add_subplot(223)
scat3 = ax3.scatter([], [], s=2, color="tab:green", alpha=0.3)
ax3.scatter(ref_RA, ref_iota, marker="x", color="black", s=50)
ax3.set_title("RA vs iota")
ax3.set_xlabel("RA")
ax3.set_ylabel(r"$\iota$")
ax3.grid(True)

# ------------------ dL vs iota -----------------
ax4 = fig.add_subplot(224)
scat4 = ax4.scatter([], [], s=2, color="tab:red", alpha=0.3)
ax4.scatter(ref_dL, ref_iota, marker="x", color="black", s=50)
ax4.set_title("dL vs iota")
ax4.set_xlabel(r"$d_L$")
ax4.set_ylabel(r"$\iota$")
ax4.grid(True)

# Fix limits once
ax2.set_xlim(ra_np.min(), ra_np.max())
ax2.set_ylim(dL_np.min(), dL_np.max())

ax3.set_xlim(ra_np.min(), ra_np.max())
ax3.set_ylim(iota_np.min(), iota_np.max())

ax4.set_xlim(dL_np.min(), dL_np.max())
ax4.set_ylim(iota_np.min(), iota_np.max())


def update(frame):

    scat1.set_offsets(np.c_[ra_np[:frame], dec_np[:frame]])
    scat2.set_offsets(np.c_[ra_np[:frame], dL_np[:frame]])
    scat3.set_offsets(np.c_[ra_np[:frame], iota_np[:frame]])
    scat4.set_offsets(np.c_[dL_np[:frame], iota_np[:frame]])

    fig.suptitle(f"Dead points: {frame}", fontsize=16)

    return scat1, scat2, scat3, scat4


anim = FuncAnimation(
    fig,
    update,
    frames=range(1, len(ra_np), 50),
    interval=50,
    blit=False,
)

plt.tight_layout()
plt.show()