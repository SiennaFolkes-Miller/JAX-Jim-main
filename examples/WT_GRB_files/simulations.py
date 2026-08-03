# Block 0: Imports and Environment Setup
import os
import jax
import jax.numpy as jnp
import blackjax
import blackjax.ns.adaptive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
import tqdm
from anesthetic import NestedSamples
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD as likelihood_function
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from astropy.time import Time
import bilby
from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    SinePrior,
    UniformPrior,
)
from anesthetic import read_chains, make_2d_axes
import jax.random as jrandom
from ripplegw import lambdas_to_lambda_tildes

# Prevents JAX from pre-allocating all available GPU memory. 
# Useful for running on shared resources.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']= 'false'