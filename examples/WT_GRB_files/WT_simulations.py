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
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
from tqdm import tqdm
from anesthetic import NestedSamples
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
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
    GaussianPrior,
)
from anesthetic import read_chains, make_2d_axes
import jax.random as jrandom

#SFM new
from jimgw.core.single_event.transform_utils import eta_to_q  
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
)

import os
output_dir = "outdir/WT"
os.makedirs(output_dir, exist_ok=True)

# Prevents JAX from pre-allocating all available GPU memory. 
# Useful for running on shared resources.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']= 'false'
# %%
# Block 1a: Main Settings
# ---------------------------
# Change these parameters to control the analysis.

# Main Settings:
analysis = "BNS"
validation = True
include_virgo = True # Set true for BNS!
n_live = 3000

# Global Settings:
fmin, fmax = 20.0, 1024.0
f_ref = 20.0
iteration_number = 3
if validation == True:
    iterations = 1
else:
    iterations = iteration_number
initial_sampling_seed = 7
mock_data_seed = 32
injection_seed = 54
sampling_frequency = 4096.0 # ONLY USED IN WAVEFORM GENERATION

# --- Nested Sampling Configuration ---
n_delete = n_live//2
threshold = 3
num_mcmc_steps_ratio = 3 # Number of MCMC steps relative to n_dims
psd_padding = 1024 # Padding for PSD estimation to avoid trigger wrap-around

# --- Prior Settings ---
lambda_min, lambda_max = 0.0, 1000.0
# Standard deviation for the conditioned (Gaussian) sky location prior
ra_std = 0.02 
dec_std = 0.01

# =========================================================
# Analysis-Specific Settings (BBH vs BNS)
# =========================================================

# Binary Black Hole Settings
if analysis == "BBH":
    # Parameters 
    columns = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec"]
    labels = [r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\iota$", r"$d_L$", r"$t_c$", r"$\phi_c$", r"$\psi$", r"$\alpha$", r"$\delta$"]
    
    tukey_alpha = 0.2

    # Timing
    fft_duration = 4.0
    post_trigger_duration = 2.0

    # Default Priors
    M_c_min, M_c_max = 10.0, 80.0
    q_min, q_max = 0.125, 1.0
    s_z_min, s_z_max = -1.0,1.0
    d_L_min, d_L_max = 100.0, 2000.0
    t_c_min, t_c_max = -0.05, 0.05
    phase_c_min, phase_c_max = 0.0, 2*jnp.pi
    psi_min, psi_max = 0.0, jnp.pi
    ra_min, ra_max = 0.0, 2*jnp.pi

    # Mock or Real?
    if validation == True:
        # GW150914
        gps = 1126259462.4
        gw150914_ra_mean = 3.446
        gw150914_dec_mean = -0.408
        gw150914_dec_std = 0.01
        gw150914_ra_std = 0.02
    else:
        # Dummy Variable 
        gps = 1130140000.0 

# Binary Neutron Stars Settings
if analysis == "BNS":
    tukey_alpha = 0.00625 # Smaller alpha for longer BNS signals
    # Parameters
    columns = ["M_c", "q", "s1_z", "s2_z", "lambda_1", "lambda_2", "iota", "d_L", "t_c", "phase_c", "psi", "ra", "dec"]
    labels = [r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\Lambda_1$",r"$\Lambda_2$",r"$\iota$", r"$d_L$", r"$t_c$", r"$\phi_c$", r"$\psi$", r"$\alpha$", r"$\delta$"]

    # Timing
    fft_duration = 128.0
    post_trigger_duration = 4.0

    # Priors
    M_c_min, M_c_max = 1.1, 2.2
    q_min, q_max = 0.125, 1.0
    s_z_min, s_z_max = -0.05, 0.05
    d_L_min, d_L_max = 10.0, 800.0
    t_c_min, t_c_max = -0.05, 0.05
    phase_c_min, phase_c_max = 0.0, 2*jnp.pi
    psi_min, psi_max = 0.0, jnp.pi
    ra_min, ra_max = 0.0, 2*jnp.pi

    # Mock or Real?
    if validation == True:
        # GW170817
        fmin, fmax = 20.0, 1024.0
        gps = 1187008882.430123
        gw170817_ra_mean = 3.446
        gw170817_dec_mean = -0.408
        gw170817_dec_std = 0.01
        gw170817_ra_std = 0.02
        include_virgo = True
        # Narrow priors for real event analysis
        d_L_min, d_L_max = 1.0, 75.0
    else:
        if include_virgo:
            gps = 1187008682.430123
        else:
            gps = 1130140000.0 

# =========================================================
# Derived Settings (do not change)
# =========================================================
n_dims = len(columns)
num_mcmc_steps = n_dims * num_mcmc_steps_ratio
gps_start = gps - fft_duration + post_trigger_duration
gps_end = gps + post_trigger_duration


# %%
# Block 1b: Store Settings in a Dictionary for Saving Later
store_global_settings = {
    "n_live": n_live,
    "n_delete": n_delete,
    "threshold": threshold,
    "num_mcmc_steps": num_mcmc_steps,
    "CBC": analysis,
    "iterations": iterations,
    "use_virgo": include_virgo,
    "validation": validation,
    "fft_duration": fft_duration,
    "post_trigger_duration": post_trigger_duration,
    "fmax": fmax,
    "fmin": fmin,
    "f_ref": f_ref,
    "initial_sampling_seed": initial_sampling_seed,
    "mock_data_seed": mock_data_seed,
    "injection_seed": injection_seed,
    "tukey_alpha": tukey_alpha,
    "psd_padding": psd_padding,
    "M_c_min": M_c_min,
    "M_c_max": M_c_max,
    "q_min": q_min,
    "q_max": q_max,
    "s_z_min": s_z_min,
    "s_z_max": s_z_max,
    "d_L_min": d_L_min,
    "d_L_max": d_L_max,
    "t_c_min": t_c_min,
    "t_c_max": t_c_max,
    "lambda_min": lambda_min,
    "lambda_max": lambda_max,
    "phase_c_min": phase_c_min,
    "phase_c_max": phase_c_max,
    "psi_min": psi_min,
    "psi_max": psi_max,
    "ra_min": ra_min,
    "ra_max": ra_max,
    "ra_std": ra_std,
    "dec_std": dec_std,    
    "gps_used": gps,       
}

# %%
# Block 2a: PSD Calculation

start_psd = int(gps) - fft_duration - 2 * psd_padding
end_psd   = int(gps) - fft_duration - psd_padding

#SFM simplified, no median filtering, had to change some variable names
print("Fetching PSD data...")
if include_virgo:
    ifos = [get_H1(), get_L1(), get_V1()]
else:
    ifos = [get_H1(), get_L1()]
for ifo in ifos:
    data = Data.from_gwosc(ifo.name, gps_start, gps_end)
    ifo.set_data(data)

    psd_data = Data.from_gwosc(ifo.name, start_psd, end_psd)
    psd_fftlength = data.duration * data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))
print("PSD calculation complete.")

psd_data = {"Frequency (Hz)": ifos[0].psd.frequencies, "H1 PSD": ifos[0].psd.values, "L1 PSD": ifos[1].psd.values}
if include_virgo:
    psd_data["V1 PSD"] = ifos[2].psd.values

psd_df = pd.DataFrame(psd_data)
#psd_df.to_csv("psd_main.csv", index=False)
#print("PSDs interpolated and saved to psd_main.csv")
# %%
# Block 3a: (if validation=True) - Fetch Real Event Data and Save

if validation == True:

    # Note: jimgw's load_data already applies the window and FFT, so we use its output directly.
    H1_strain = ifos[0].data.fd
    L1_strain = ifos[1].data.fd
    H1_freqs = ifos[0].data.frequencies
    if include_virgo:
        V1_strain = ifos[2].data.fd
        
    if analysis == "BBH":
        filename = "GW150914.csv"
    elif analysis == "BNS":
        filename = "GW170817.csv"
        
    # Save FFT strain to CSV
    strain_data = {
        "Frequency (Hz)": H1_freqs,
        "H1 Strain": [str(x) for x in np.array(H1_strain)],
        "L1 Strain": [str(x) for x in np.array(L1_strain)],
    }
    if include_virgo:
        strain_data["V1 Strain"] = [str(x) for x in np.array(V1_strain)]
        
    df = pd.DataFrame(strain_data)
    #df.to_csv(filename, index=False)
    #print(f"Real {analysis} event data saved as: {filename}")

# %%
# Block 3b: (if validation=False) - Simulate Signal, Inject into Noise, and Save

jax.config.update("jax_enable_x64", True)

#SFM new conditional prior
def build_conditional_prior(ra_mean, ra_std, dec_mean, dec_std):
    if analysis == "BBH":
        prior_C = CombinePrior([
            UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
            UniformPrior(q_min,q_max, parameter_names=["q"]),
            UniformPrior(s_z_min,s_z_max, parameter_names=["s1_z"]),
            UniformPrior(s_z_min,s_z_max, parameter_names=["s2_z"]),
            SinePrior(parameter_names=["iota"]),
            UniformPrior(d_L_min, d_L_max, parameter_names=["d_L"]),
            UniformPrior(t_c_min, t_c_max, parameter_names=["t_c"]),
            UniformPrior(phase_c_min, phase_c_max, parameter_names=["phase_c"]),
            UniformPrior(psi_min, psi_max, parameter_names=["psi"]),
            GaussianPrior(mu=ra_mean, sigma=ra_std, parameter_names=["ra"]),
            GaussianPrior(mu=dec_mean, sigma=dec_std, parameter_names=["dec"]),
        ])
    elif analysis == "BNS":
            prior_C = CombinePrior([
            UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
            UniformPrior(q_min,q_max, parameter_names=["q"]),
            UniformPrior(s_z_min,s_z_max, parameter_names=["s1_z"]),
            UniformPrior(s_z_min,s_z_max, parameter_names=["s2_z"]),
            SinePrior(parameter_names=["iota"]),
            #SFM removed NOTE: Using a narrow d_L prior to ensure a high SNR injection for this analysis.
            UniformPrior(d_L_min,d_L_max, parameter_names=["d_L"]),
            #PowerLawPrior(d_L_min, d_L_max, 3.0, parameter_names=["d_L"]),
            UniformPrior(t_c_min, t_c_max, parameter_names=["t_c"]),
            UniformPrior(phase_c_min, phase_c_max, parameter_names=["phase_c"]),
            UniformPrior(psi_min, psi_max, parameter_names=["psi"]),
            GaussianPrior(mu=ra_mean, sigma=ra_std, parameter_names=["ra"]),
            GaussianPrior(mu=dec_mean, sigma=dec_std, parameter_names=["dec"]),
            UniformPrior(lambda_min,lambda_max, parameter_names=["lambda_1"]),
            UniformPrior(lambda_min,lambda_max, parameter_names=["lambda_2"]),
        ])
    return prior_C

if analysis == "BBH":
    prior_U = CombinePrior([
        UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
        UniformPrior(q_min,q_max, parameter_names=["q"]),
        UniformPrior(s_z_min,s_z_max, parameter_names=["s1_z"]),
        UniformPrior(s_z_min,s_z_max, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        UniformPrior(d_L_min, d_L_max, parameter_names=["d_L"]),
        UniformPrior(t_c_min, t_c_max, parameter_names=["t_c"]),
        UniformPrior(phase_c_min, phase_c_max, parameter_names=["phase_c"]),
        UniformPrior(psi_min, psi_max, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
    ])
elif analysis == "BNS":
    prior_U = CombinePrior([
        UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"]),
        UniformPrior(q_min,q_max, parameter_names=["q"]),
        UniformPrior(s_z_min,s_z_max, parameter_names=["s1_z"]),
        UniformPrior(s_z_min,s_z_max, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        #SFM removed NOTE: Using a narrow d_L prior to ensure a high SNR injection for this analysis.
        UniformPrior(d_L_min,d_L_max, parameter_names=["d_L"]),

        UniformPrior(t_c_min, t_c_max, parameter_names=["t_c"]),
        UniformPrior(phase_c_min, phase_c_max, parameter_names=["phase_c"]),
        UniformPrior(psi_min, psi_max, parameter_names=["psi"]),
        UniformPrior(ra_min, ra_max, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(lambda_min,lambda_max, parameter_names=["lambda_1"]),
        UniformPrior(lambda_min,lambda_max, parameter_names=["lambda_2"]),
    ])

if validation == False:
    truth_values = []
    injection = []
    SNR = []
    rng_key = jax.random.PRNGKey(mock_data_seed)

    for i in range(iterations):
        print(f"\n--- Generating Mock Injection {i+1}/{iterations} ---")
        rng_key, subkey = jax.random.split(rng_key)
        true_params = prior_U.sample(subkey, 1)
        true_params = {k: float(v.squeeze()) for k, v in true_params.items()} 
        M_c = true_params["M_c"]
        q = true_params["q"]
        eta = q / ((1 + q) ** 2)  # Symmetric mass ratio
        M_total = M_c / (eta ** (3/5))
        mass_2 = M_total / (1 + q)
        mass_1 = M_total - mass_2
        geocent_time = gps + true_params["t_c"]
        
        if analysis == "BBH":
            true_value_iteration = {k: true_params.get(k, np.nan) for k in columns}
            injection_iteration = {
                "mass_1": mass_1, "mass_2": mass_2,
                "chi_1": true_params["s1_z"], "chi_2": true_params["s2_z"], 
                "theta_jn": true_params["iota"], "luminosity_distance": true_params["d_L"],
                "phase": true_params["phase_c"], "psi": true_params["psi"],
                "geocent_time": geocent_time, "ra": true_params["ra"], "dec": true_params["dec"],
            }
        elif analysis == "BNS":
            true_value_iteration = {k: true_params.get(k, np.nan) for k in columns}
            injection_iteration = {
                "mass_1": mass_1, "mass_2": mass_2,
                "chi_1": true_params["s1_z"], "chi_2": true_params["s2_z"], 
                "theta_jn": true_params["iota"], "luminosity_distance": true_params["d_L"],
                "phase": true_params["phase_c"], "psi": true_params["psi"],
                "geocent_time": geocent_time, "ra": true_params["ra"], "dec": true_params["dec"],
                "lambda_1": true_params["lambda_1"], "lambda_2": true_params["lambda_2"],
            }
            
        print(f"Injection parameters (for (not) bilby):\n{injection_iteration}")
        truth_values.append(true_value_iteration)
        injection.append(injection_iteration)

    df_true_values = pd.DataFrame(truth_values)
    injection_parameter_df = pd.DataFrame(injection)
    truth_path = os.path.join(output_dir, "truth_metadata_main.csv")
    df_true_values.to_csv(truth_path, index=False)
    injection_path = os.path.join(output_dir,"injection_parameters_all_main.csv")
    injection_parameter_df.to_csv(injection_path, index=False)
    print("\nCompleted Parameter Sampling. Generating waveforms...")

    if analysis == "BBH":
        waveform = RippleIMRPhenomD(f_ref=f_ref)
    else:
        waveform = RippleIMRPhenomD_NRTidalv2(f_ref=f_ref)

    for i in range(len(injection_parameter_df)):
        print(f"\nGenerating and injecting waveform {i+1}/{iterations}")
        injection_params = injection_parameter_df.iloc[i].to_dict()

        params = {
            "M_c": df_true_values.iloc[i]["M_c"],
            "q": df_true_values.iloc[i]["q"],
            "eta": df_true_values.iloc[i]["q"] / (1 + df_true_values.iloc[i]["q"])**2,
            "s1_z": injection_params["chi_1"],
            "s2_z": injection_params["chi_2"],
            "iota": injection_params["theta_jn"],
            "d_L": injection_params["luminosity_distance"],
            "t_c": injection_params["geocent_time"] - gps,
            "phase_c": injection_params["phase"],
            "psi": injection_params["psi"],
            "ra": injection_params["ra"],
            "dec": injection_params["dec"],
            "trigger_time": injection_params["geocent_time"],
            "gmst": Time(injection_params["geocent_time"], format="gps").sidereal_time("apparent","greenwich").rad,
        }

        if analysis == "BNS":
            params["lambda_1"] = injection_params["lambda_1"]
            params["lambda_2"] = injection_params["lambda_2"]

        # Use jimgw to handle noise realization and injection
        H1 = ifos[0]
        L1 = ifos[1]
        if include_virgo:
            V1 = ifos[2]
        
        # Inject into jimgw detector objects
        sky_params = {
            "ra": injection_params["ra"], "dec": injection_params["dec"], "psi": injection_params["psi"],
            "gmst": Time(injection_params["geocent_time"], format="gps").sidereal_time("apparent","greenwich").rad,
            "epoch": fft_duration-post_trigger_duration,
            "t_c": injection_params["geocent_time"] - gps
        }
        waveform_pols = waveform(H1.frequencies, params)
        h_sky = {
            "p": waveform_pols["p"],
            "c": waveform_pols["c"]
        }

        rng_key, sk1, sk2, sk3 = jrandom.split(rng_key, 4)
        H1.inject_signal(
            duration=fft_duration,
            sampling_frequency=H1.data.sampling_frequency,
            trigger_time=injection_params["geocent_time"],
            waveform_model=waveform,
            parameters=params,
            f_min=fmin,
            f_max=fmax,
            rng_key=sk1,
        )

        L1.inject_signal(
            duration=fft_duration,
            sampling_frequency=L1.data.sampling_frequency,
            trigger_time=injection_params["geocent_time"],
            waveform_model=waveform,
            parameters=params,
            f_min=fmin,
            f_max=fmax,
            rng_key=sk2,
        )

        if include_virgo:
            V1.inject_signal(
                duration=fft_duration,
                sampling_frequency=V1.data.sampling_frequency,
                trigger_time=injection_params["geocent_time"],
                waveform_model=waveform,
                parameters=params,
                f_min=fmin,
                f_max=fmax,
                rng_key=sk3,
            )
        #SFM saving SNR
        H1_snr = float(H1.optimal_snr)
        L1_snr = float(L1.optimal_snr)

        network_snr = np.sqrt(H1_snr**2 + L1_snr**2)

        snr_entry = {
            "H1": H1_snr,
            "L1": L1_snr,
            "Network": network_snr,
            "H1_matched_filter": complex(H1.match_filtered_snr),
            "L1_matched_filter": complex(L1.match_filtered_snr),
        }

        if include_virgo:
            V1_snr = float(V1.optimal_snr)
            snr_entry["V1"] = V1_snr
            snr_entry["V1_matched_filter"] = complex(V1.match_filtered_snr)
            snr_entry["Network"] = np.sqrt(
                H1_snr**2 + L1_snr**2 + V1_snr**2
            )

        SNR.append(snr_entry)
        # Save strain data to file
        strain_data = {
            "Frequency (Hz)": H1.frequencies,
            "H1 Strain": [str(x) for x in np.array(H1.data.fd)],
            "L1 Strain": [str(x) for x in np.array(L1.data.fd)],
        }
        if include_virgo:
            strain_data["V1 Strain"] = [str(x) for x in np.array(V1.data.fd)]
        
        df = pd.DataFrame(strain_data)
        #filename = f"{analysis}_waveform_output_{i}_main.csv"
        #df.to_csv(filename, index=False)
        #print(f"{analysis} mock data saved to: {filename}")
        

    df_SNR = pd.DataFrame(SNR)

# %%
# Block 4: Main Sampling Execution (Refactored)

jax.config.update("jax_enable_x64", True)


#SFM new transforms
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

sample_transforms, likelihood_transforms = build_transforms(gps, ifos)

#SFM needs editing
if analysis == "BBH":
    ref_param = {
        "M_c": 3.10497857e01,
        "q": float(eta_to_q(0.15874815)),
        "s1_z": 0.5,
        "s2_z": 0.5,
        "d_L": 5.47223231e02,
        "t_c": 1.29378808e-02,
        "phase_c": 3.30994042e00,
        "iota": 1.17146435,
        "psi": 3.41074151e-02,
        "ra": 2.55345319e00,
        "dec": -1.26006121,
    }
else: #BNS
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
        'ra': 3.39736826,
        'dec': -0.34000186
    }


ref_param = prepare_ref_params(ref_param, likelihood_transforms)
#end of SFM additions





# Define global variables needed in functions
gmst = float(Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad)
epoch = fft_duration - post_trigger_duration

if validation:
    iterations = 1

#SFM new
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

#end of new SFM additions






for i in range(iterations):
    # --- Setup for this iteration ---
    if analysis == "BBH":
        waveform = RippleIMRPhenomD(f_ref=f_ref)
    else: # BNS
        waveform = RippleIMRPhenomD_NRTidalv2(f_ref=f_ref)

    likelihood_U = HeterodynedTransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        trigger_time=gps,
        f_min=fmin,
        f_max=fmax,
        n_bins=256,   #goal is 501
        prior=prior_U,
        #reference_parameters=ref_param,
        optimizer_popsize=10,
        optimizer_n_steps=50,   #goal is 100
        likelihood_transforms=likelihood_transforms,
        phase_marginalization=True,
    )
    def loglikelihood_U(x):
        for transform in reversed(sample_transforms):
            x, _ = transform.inverse(x)
        for transform in reversed(likelihood_transforms):
            x = transform.forward(x)
        like = likelihood_U.evaluate(x)  
        return like

    # --- Run Unconditioned Analysis --- SFM heavily updated
    print("\n" + "-"*58 + f"\nRunning Unconditioned Sampling (Iteration {i+1})\n" + "-"*58)
    output_path_U = os.path.join(output_dir,f"samples_{analysis}_unconditioned_{i}_main.csv")
    
    # Initialize particles from the unconditioned prior
    rng_key = jax.random.PRNGKey(initial_sampling_seed)
    rng_key, init_key = jax.random.split(rng_key, 2)
    initial_particles_U = prior_U.sample(init_key, n_live)
    initial_particles_U = sample_to_unbound(initial_particles_U)
    
    
    # Run the sampler for the unconditioned case...
    nested_sampler_U = blackjax.nss(
            logprior_fn=logprior_unconditioned, 
            loglikelihood_fn=loglikelihood_U,
            num_delete=n_delete, 
            num_inner_steps=num_mcmc_steps)

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
    with tqdm(desc="Unconditioned Dead Points", unit=" points") as pbar:
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
        labels=labels,
    )
    nss_dataframe_U.to_csv(output_path_U)
    print(f"Unconditioned run complete. Saved to {output_path_U}")

    # --- Run Conditioned Analysis --- SFM heavily updated
    print("\n" + "-"*58 + f"\nRunning Conditioned Sampling (Iteration {i+1})\n" + "-"*58)
    output_path_C = os.path.join(output_dir,f"samples_{analysis}_conditioned_{i}_main.csv")

    if validation:
        ra_mean = gw170817_ra_mean if analysis == "BNS" else gw150914_ra_mean
        dec_mean = gw170817_dec_mean if analysis == "BNS" else gw150914_dec_mean
    else:
        true_vals = df_true_values.iloc[i]
        ra_mean, dec_mean = true_vals['ra'], true_vals['dec']

    prior_C = build_conditional_prior(ra_mean, ra_std, dec_mean, dec_std)

    likelihood_C = HeterodynedTransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        trigger_time=gps,
        f_min=fmin,
        f_max=fmax,
        n_bins=256,   #goal is 501
        prior=prior_C,
        reference_parameters=ref_param,
        optimizer_popsize=10,
        optimizer_n_steps=50,   #goal is 100
        likelihood_transforms=likelihood_transforms,
        phase_marginalization=True,
    )
    def loglikelihood_C(x):
        for transform in reversed(sample_transforms):
            x, _ = transform.inverse(x)
        for transform in reversed(likelihood_transforms):
            x = transform.forward(x)
        like = likelihood_C.evaluate(x)  
        return like

    # Initialize particles from the conditioned prior
    rng_key, init_key = jax.random.split(rng_key, 2)
    initial_particles_C = prior_C.sample(init_key, n_live)
    initial_particles_C = sample_to_unbound(initial_particles_C)
    
    # Run the sampler for the conditioned case...
    nested_sampler_C = blackjax.nss(
            logprior_fn=logprior_conditioned, 
            loglikelihood_fn=loglikelihood_U,
            num_delete=n_delete, 
            num_inner_steps=num_mcmc_steps)
    
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
        labels=labels,
    )
    nss_dataframe_C.to_csv(output_path_C)
    print(f"Conditioned run complete. Saved to {output_path_C}")

# %%
# Block 5.1: Log-Evidence Comparison

for i in range(iterations):
    samplesU_path = os.path.join(
        output_dir,
        f"samples_{analysis}_unconditioned_{i}_main.csv"
    )

    samplesC_path = os.path.join(
        output_dir,
        f"samples_{analysis}_conditioned_{i}_main.csv"
    )

    loaded_samplesU = read_chains(samplesU_path)
    logZ_samplesU = loaded_samplesU.logZ(100)
    logZ_meanU, logZ_stdU = logZ_samplesU.mean(), logZ_samplesU.std()
    
    loaded_samplesC = read_chains(samplesC_path)
    logZ_samplesC = loaded_samplesC.logZ(100)
    logZ_meanC, logZ_stdC = logZ_samplesC.mean(), logZ_samplesC.std()

    #SFM added
    weights_U = loaded_samplesU.get_weights()
    weights_C = loaded_samplesC.get_weights()

    print("Unconditioned:")
    for param in ["d_L", "ra", "dec"]:
        print(f"{param} = {np.average(getattr(loaded_samplesU, param), weights=weights_U)}")

    print("\nConditioned:")
    for param in ["d_L", "ra", "dec"]:
        print(f"{param} = {np.average(getattr(loaded_samplesC, param), weights=weights_C)}")

    
    
    print(f"--- Results for Iteration {i} ---")
    print(f"Unconditioned LogZ Estimate: {logZ_meanU} ± {logZ_stdU}")
    print(f"Conditioned LogZ Estimate:   {logZ_meanC} ± {logZ_stdC}")

# %%
# Block 5.2: Final Metadata Aggregation

for i in range(iterations):
    if not validation:
        sample_folder_C = os.path.join(
            output_dir,
            f"samples_{analysis}_conditioned_{i}_main.csv"
        )
        sample_folder_U = os.path.join(
            output_dir,
            f"samples_{analysis}_unconditioned_{i}_main.csv"
        )

    elif validation and analysis == "BNS":
        sample_folder_U = os.path.join(output_dir, "samples_BNS_unconditioned_0_main.csv")
        sample_folder_C = os.path.join(output_dir, "samples_BNS_conditioned_0_main.csv")

    else:  # validation and BBH
        sample_folder_U = os.path.join(output_dir, "samples_BBH_unconditioned_0_main.csv")
        sample_folder_C = os.path.join(output_dir, "samples_BBH_conditioned_0_main.csv")

    loaded_sample_C = read_chains(sample_folder_C)
    loaded_sample_U = read_chains(sample_folder_U)

    logZ_samplesC = loaded_sample_C.logZ(100)
    logZ_meanC, logZ_stdC = logZ_samplesC.mean(), logZ_samplesC.std()

    logZ_samplesU = loaded_sample_U.logZ(100)
    logZ_meanU, logZ_stdU = logZ_samplesU.mean(), logZ_samplesU.std()

    output_dict = {
        "iteration": i,
        "analysis": analysis,
        "logZ_unconditioned": logZ_meanU,
        "logZ_std_unconditioned": logZ_stdU,
        "logZ_conditioned": logZ_meanC,
        "logZ_std_conditioned": logZ_stdC,
    }
    
    # Add injection/SNR data if it exists (i.e., not validation run)
    if not validation:
        output_dict["injection_parameters"] = injection_parameter_df.iloc[i].to_dict()
        output_dict["true_parameters"] = df_true_values.iloc[i].to_dict()
        output_dict["SNR"] = df_SNR.iloc[i].to_dict()
    else:
        output_dict["injection_parameters"] = "real_data"
        output_dict["true_parameters"] = "real_data"
        output_dict["SNR"] = "real_data"
        
    output_dict["settings"] = store_global_settings

    # Flatten nested dicts for clean CSV output
    flat_output = {}
    for k, v in output_dict.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat_output[f"{k}.{subk}"] = subv
        else:
            flat_output[k] = v

    # Save to the correct filename
    if validation:
        filename = os.path.join(
            output_dir,
            "GW170817_data_main.csv" if analysis == "BNS" else "GW150914_data_main.csv"
        )
    else:
        filename = os.path.join(
            output_dir,
            f"{analysis}_{i}_data_main.csv"
        )

    pd.DataFrame([flat_output]).to_csv(filename, index=False)
    print(f"Saved metadata to: {filename}")

print("All data saved.")