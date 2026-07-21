
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

    prior.extend([RA_prior, Dec_prior])
    return CombinePrior(prior)


#sample prior
prior = build_prior()


rng_key = jax.random.PRNGKey(42)
num_live = 10000
num_delete = num_live//2


samples = prior.sample(rng_key, num_live)


ra = samples["ra"]
dec = samples["dec"]
# convert to numpy for matplotlib
ra = jnp.asarray(ra) - jnp.pi
dec = jnp.asarray(dec)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection="mollweide")  #also could use aitoff or hammer as well

#histogram and scatter
hist = ax.hist2d(ra,dec,bins=50,cmap="viridis", alpha=0.5)
plt.colorbar(hist[3], ax=ax, label="Number of particles")
ax.scatter(ra,dec,s=1,color='k',alpha=0.2)

ax.grid(True)
ax.set_xlabel("RA")
ax.set_ylabel("Dec")

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

ra = jnp.asarray(nss_samples["ra"]) - jnp.pi
dec = jnp.asarray(nss_samples["dec"])



from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection="mollweide")

scat = ax.scatter([], [], s=2, color="m", alpha=0.3)
#ax.scatter(ref_RA, ref_Dec, marker='x', s=10)

ax.grid(True)
ax.set_xlabel("RA")
ax.set_ylabel("Dec")


def update(frame):
    scat.set_offsets(jnp.column_stack((ra[:frame], dec[:frame])))
    ax.set_title(f"Dead points: {frame}")
    return scat,

anim = FuncAnimation(
    fig,
    update,
    frames=range(1, len(ra), 50),   #every 'x' deadpoints
    interval=50,  #seconds between frames
    blit=False,
)

plt.show()



