# %%
# Block 0: Imports and Environment Setup
import os
import jax
import jax.numpy as jnp
import blackjax
import blackjax.ns.adaptive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
from tqdm import tqdm
from anesthetic import NestedSamples
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from anesthetic import read_chains
import jax.random as jrandom


from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    PowerLawPrior,
    SinePrior,
    UniformPrior,
    GaussianPrior,
)

from jimgw.core.single_event.transform_utils import eta_to_q  
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
)

# Configure JAX for 64-bit precision and memory management
os.environ["JAX_ENABLE_X64"] = "True"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']= 'false'

# %%
# Block 1a: Global Configuration

# --- Sampling Settings ---
n_live = 200
initial_sampling_seed = 34
n_delete = n_live // 2
threshold = 1
num_mcmc_steps_ratio = 1

# --- Data and Signal Processing Settings ---
fmin, fmax = 20.0, 800.0
f_ref = 20.0
fft_duration = 128.0
post_trigger_duration = 4.0
psd_padding = 500  # Duration for PSD estimation to avoid edge effects
tukey_alpha = 0.00625 # Windowing parameter to reduce spectral leakage
epoch = fft_duration - post_trigger_duration

# --- Prior Parameter Ranges ---
M_c_min, M_c_max = 1.1, 2.2
q_min, q_max = 0.125, 1.0
s_z_min, s_z_max = -0.05, 0.05
lambda_min, lambda_max = 0.0, 1000.0
d_L_min, d_L_max = 20.0, 800.0
t_c_min, t_c_max = -1.0, 1.0  # Search window around GRB time
phase_c_min, phase_c_max = 0.0, 2 * jnp.pi
psi_min, psi_max = 0.0, jnp.pi
ra_min, ra_max = 0.0, 2 * jnp.pi

# -- Conditioned Prior Settings ---
ra_std = 0.2  # Std deviation for Gaussian prior on RA (radians)
dec_std = 0.1 # Std deviation for Gaussian prior on DEC (radians)

# --- Parameter Definitions for Sampler ---
columns = ["M_c", "q", "s1_z", "s2_z", "lambda_1", "lambda_2", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec"]
labels = [r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\Lambda_1$",r"$\Lambda_2$",r"$\iota$", r"$d_L$", r"$t_c$", r"$\phi_c$", r"$\psi$", r"$\alpha$", r"$\delta$"]

# --- Derived Settings ---
n_dims = len(columns)
num_mcmc_steps = n_dims * num_mcmc_steps_ratio

# %%
# Block 1b: sGRB Catalog Data

grb_data_raw = [
    ["GRB190724031", "11 21 24.0", "+15 09 00", "2019-07-24", "00:43:57", 0.080],
    ["GRB190525032", "22 32 04.8", "+05 27 00", "2019-05-25", "00:45:48", 0.896],
    ["GRB190728271", "23 46 45.6", "+05 25 48", "2019-07-28", "06:30:37", 0.832],
    ["GRB190813520", "07 05 31.2", "-23 16 12", "2019-08-13", "12:29:10", 0.192],
    ["GRB190510430", "08 32 31.2", "+33 33 00", "2019-05-10", "10:19:16", 0.640],
]

def process_grb_data(raw_data):
    """
    Converts raw GRB data (strings) into a structured pandas DataFrame
    with GPS times and sky coordinates in radians.
    """
    results = []
    for name, ra_str, dec_str, date_str, time_str, t90 in raw_data:
        # Convert UTC to GPS time
        utc_str = f"{date_str}T{time_str}"
        gps_time = Time(utc_str, format="isot", scale="utc").gps

        # Convert RA/DEC to radians
        skycoord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        
        results.append({
            "GRB": name,
            "GPS": gps_time,
            "RA_rad": skycoord.ra.radian,
            "DEC_rad": skycoord.dec.radian,
            "T90_s": t90,
        })
    return pd.DataFrame(results)

grb_catalog = process_grb_data(grb_data_raw)
print("Processed GRB Catalog:")
print(grb_catalog)

# %%
# Block 2a: Helper Functions for Analysis

#SFM additions
def build_transforms(gps_time, ifos):
    sample_transforms = [GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time=gps_time, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps_time, ifos=ifos)]
    likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]
    return sample_transforms, likelihood_transforms

def prepare_ref_params(ref_param: dict, likelihood_transforms):
    """Forward-map reference parameters through the likelihood transforms."""
    for transform in reversed(likelihood_transforms):
        ref_param = transform.forward(ref_param)
    return ref_param

prior_U = CombinePrior([
    UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
    UniformPrior(q_min, q_max, parameter_names=["q"]),
    UniformPrior(s_z_min, s_z_max, parameter_names=["s1_z"]),
    UniformPrior(s_z_min, s_z_max, parameter_names=["s2_z"]),
    SinePrior(parameter_names=["iota"]),
    UniformPrior(d_L_min, d_L_max, parameter_names=["d_L"]),
    UniformPrior(t_c_min, t_c_max, parameter_names=["t_c"]),
    UniformPrior(phase_c_min, phase_c_max, parameter_names=["phase_c"]),
    UniformPrior(psi_min, psi_max, parameter_names=["psi"]),
    UniformPrior(ra_min, ra_max, parameter_names=["ra"]),
    CosinePrior(parameter_names=["dec"]),
    UniformPrior(lambda_min, lambda_max, parameter_names=["lambda_1"]),
    UniformPrior(lambda_min, lambda_max, parameter_names=["lambda_2"]),
])

def build_conditional_prior(ra_mean, dec_mean, ra_std, dec_std):

    prior_C = CombinePrior([
        UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
        UniformPrior(q_min, q_max, parameter_names=["q"]),
        UniformPrior(s_z_min, s_z_max, parameter_names=["s1_z"]),
        UniformPrior(s_z_min, s_z_max, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        UniformPrior(d_L_min, d_L_max, parameter_names=["d_L"]),
        UniformPrior(t_c_min, t_c_max, parameter_names=["t_c"]),
        UniformPrior(phase_c_min, phase_c_max, parameter_names=["phase_c"]),
        UniformPrior(psi_min, psi_max, parameter_names=["psi"]),
        GaussianPrior(mu=ra_mean, sigma=ra_std, parameter_names=["ra"]),
        GaussianPrior(mu=dec_mean, sigma=dec_std, parameter_names=["dec"]),
        UniformPrior(lambda_min, lambda_max, parameter_names=["lambda_1"]),
        UniformPrior(lambda_min, lambda_max, parameter_names=["lambda_2"]),
    ])

    return prior_C


def logprior_unconditioned(x):
    transform_jacobian = 0.0
    for transform in reversed(sample_transforms):
        x, jacobian = transform.inverse(x)
        transform_jacobian += jacobian
    return prior_U.log_prob(x) + transform_jacobian

def logprior_conditioned(x):
    transform_jacobian = 0.0
    for transform in reversed(sample_transforms):
        x, jacobian = transform.inverse(x)
        transform_jacobian += jacobian
    return prior_C.log_prob(x) + transform_jacobian

def sample_to_unbound(x):
    for t in sample_transforms:
            x = jax.vmap(t.forward)(x)
    return x

def unbound_to_sample(x):
    for t in reversed(sample_transforms):
        x, _ = t.inverse(x)
    return x


def process_samples_to_physical(unbounded_samples):
    """Convert unbounded samples to physical parameter space."""
    sample_dict = {key: np.array(value) for key, value in unbounded_samples.items()}
    physical_samples = sample_dict.copy()
    for transform in reversed(likelihood_transforms):
        inputs = {name: physical_samples[name] for name in transform.name_mapping[0]}
        outputs = transform.transform_func(inputs)
        physical_samples.update({k: np.array(v) for k, v in outputs.items()})
    physical_samples.setdefault("q", sample_dict.get("q"))
    physical_samples.pop("eta", None)
    return physical_samples



# %%
all_results = []
for i, (_, grb_info) in enumerate(grb_catalog.iterrows(), start=1):
    """
    Main pipeline function. Takes a dictionary with GRB info, runs both
    unconditioned and conditioned nested sampling, and returns the results.
    """
    try:
        grb_name = grb_info["GRB"]
        gps_time = grb_info["GPS"]
        ra_mean = grb_info["RA_rad"]
        dec_mean = grb_info["DEC_rad"]

        print(f"\n{'='*60}")
        print(f"Analyzing {grb_name} ({i}/{len(grb_catalog)})")
        print(f"{'='*60}")

        # --- 1. Load and prepare data ---

        gps_start = gps_time - fft_duration + post_trigger_duration
        gps_end = gps_time + post_trigger_duration

        start_psd = int(gps_time) - fft_duration - 2 * psd_padding
        end_psd = int(gps_time) - fft_duration - psd_padding


        print("Fetching and processing detector data...")
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            data = Data.from_gwosc(ifo.name, gps_start, gps_end)
            ifo.set_data(data)

            psd_data = Data.from_gwosc(ifo.name, start_psd, end_psd)
            psd_fftlength = data.duration * data.sampling_frequency
            ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))

        print("PSD calculation complete.")

        frequencies = ifos[0].data.frequencies
        gmst = float(Time(gps_time, format="gps").sidereal_time("apparent", "greenwich").rad)
        sample_transforms, likelihood_transforms = build_transforms(gps_time, ifos)

        ref_param = {
            'M_c': 1.1975896,
            'q': float(eta_to_q(0.2461001)),  
            's1_z': -0.01890608,
            's2_z': 0.04888488,
            'lambda_1': 791.04366468,
            'lambda_2': 891.04366468,
            'd_L': 40.06331818,
            't_c': 0.00193536,
            'phase_c': 5.88649652,
            'iota': 1.93095421,
            'psi': 1.59687217,
            'ra': ra_mean,
            'dec': dec_mean}

        ref_param = prepare_ref_params(ref_param, likelihood_transforms)
        
        # --- 2. Define Likelihood and Waveform ---
        waveform = RippleIMRPhenomD_NRTidalv2(f_ref=f_ref)
        print("\n" + "-"*58)
        print("Running Unconditioned Sampling")
        print("-"*58)

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            trigger_time=gps_time,
            f_min=fmin,
            f_max=fmax,
            n_bins=256,
            prior=prior_U,
            reference_parameters=ref_param,
            optimizer_popsize=10,
            optimizer_n_steps=50,
            likelihood_transforms=likelihood_transforms,
            phase_marginalization=True,
        )
        def loglikelihood_fn(x):
            for transform in reversed(sample_transforms):
                x, _ = transform.inverse(x)
            for transform in reversed(likelihood_transforms):
                x = transform.forward(x)
            like = likelihood.evaluate(x)  
            return like
        
        # --- 3. Run Unconditioned and Conditioned Sampling ---
        rng_key = jax.random.PRNGKey(initial_sampling_seed)
        rng_key, init_key = jax.random.split(rng_key)

        initial_particles_U = prior_U.sample(init_key, n_live)
        initial_particles_U = sample_to_unbound(initial_particles_U)
        nested_sampler_U = blackjax.nss(
            logprior_fn=logprior_unconditioned,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=n_delete,
            num_inner_steps=num_mcmc_steps,
        )

        @jax.jit
        def one_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = nested_sampler_U.step(subk, state)
            return (state, k), dead_point


        state = nested_sampler_U.init(initial_particles_U)
        (state_dummy, rng_key), _ = one_step((state, rng_key), None)
        jax.block_until_ready(state_dummy.integrator.logZ) 

        dead = []
        with tqdm(desc=f"Unconditioned Dead Points", unit=" points") as pbar:
            while not state.integrator.logZ_live - state.integrator.logZ < -1 *threshold: 
                rng_key, step_key = jax.random.split(rng_key)
                state, dead_info = jax.jit(nested_sampler_U.step)(step_key, state)
                dead.append(dead_info)
                pbar.update(n_delete)

        samples_U = blackjax.ns.utils.finalise(state, dead)
        nss_unbounded_U = jax.vmap(unbound_to_sample)(samples_U.particles.position)
        nss_physical_U = process_samples_to_physical(nss_unbounded_U)
        nss_dataframe_U = NestedSamples(
            data=nss_physical_U,
            logL=samples_U.particles.loglikelihood,    
            logL_birth=samples_U.particles.loglikelihood_birth, 
            columns=columns, 
            labels=labels,
        )
        output_path_U = f"samples_{grb_name}_unconditioned.csv"
        nss_dataframe_U.to_csv(output_path_U)
        print(f"Unconditioned run complete. Saved to {output_path_U}")
            
        # Calculate logZ
        logZ_U = nss_dataframe_U.logZ(100)
        logZ_U_mean = logZ_U.mean()
        logZ_U_std = logZ_U.std()

        print("\n" + "-"*58)
        print("Running Conditioned Sampling")
        print("-"*58)

        prior_C = build_conditional_prior(ra_mean,dec_mean, ra_std, dec_std)

        def loglikelihood_fn(x):
            for transform in reversed(sample_transforms):
                x, _ = transform.inverse(x)
            for transform in reversed(likelihood_transforms):
                x = transform.forward(x)
            return likelihood.evaluate(x)

        rng_key, init_key = jax.random.split(rng_key)

        initial_particles_C = prior_C.sample(init_key, n_live)
        initial_particles_C = sample_to_unbound(initial_particles_C)

        nested_sampler_C = blackjax.nss(
            logprior_fn=logprior_conditioned,
            loglikelihood_fn=loglikelihood_fn,
            num_delete=n_delete,
            num_inner_steps=num_mcmc_steps,
        )

        @jax.jit
        def one_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = nested_sampler_C.step(subk, state)
            return (state, k), dead_point

        state = nested_sampler_C.init(initial_particles_C)
        (state_dummy, rng_key), _ = one_step((state, rng_key), None)
        jax.block_until_ready(state_dummy.integrator.logZ)

        dead = []
        with tqdm(desc="Conditioned Dead Points", unit=" points") as pbar:
            while not state.integrator.logZ_live - state.integrator.logZ < -1 * threshold:
                rng_key, step_key = jax.random.split(rng_key)
                state, dead_info = jax.jit(nested_sampler_C.step)(step_key, state)
                dead.append(dead_info)
                pbar.update(n_delete)

        samples_C = blackjax.ns.utils.finalise(state, dead)
        nss_unbounded_C = jax.vmap(unbound_to_sample)(samples_C.particles.position)
        nss_physical_C = process_samples_to_physical(nss_unbounded_C)
        nss_dataframe_C = NestedSamples(
            data=nss_physical_C,
            logL=samples_C.particles.loglikelihood,
            logL_birth=samples_C.particles.loglikelihood_birth,
            columns=columns,
            labels=labels,
        )

        output_path_C = f"samples_{grb_name}_conditioned.csv"
        nss_dataframe_C.to_csv(output_path_C)

        print(f"Conditioned run complete. Saved to {output_path_C}")

        logZ_C = nss_dataframe_C.logZ(100)
        logZ_C_mean = logZ_C.mean()
        logZ_C_std = logZ_C.std()

        result = {
            "GRB": grb_name,
            "GPS": gps_time,
            "RA_rad": ra_mean,
            "DEC_rad": dec_mean,
            "T90_s": grb_info["T90_s"],
            "logZ_unconditioned": logZ_U_mean,
            "logZ_std_unconditioned": logZ_U_std,
            "logZ_conditioned": logZ_C_mean,
            "logZ_std_conditioned": logZ_C_std,
            "delta_logZ": logZ_C_mean - logZ_U_mean,
            "delta_logZ_err": np.sqrt(logZ_U_std**2 + logZ_C_std**2),
        }

        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            "grb_analysis_summary.csv",
            index=False,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        continue



# %%
# Block 4: Main Execution Loop

results_df = pd.DataFrame(all_results)
results_df.to_csv("grb_analysis_summary.csv", index=False)

print(results_df)
