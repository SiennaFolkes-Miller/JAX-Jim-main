
# # ############################
# # ##copied GW150914 pipeline##
# # ############################


import argparse
import json
import os
import anesthetic
from anesthetic import NestedSamples
import blackjax
from blackjax.smc.ess import ess as smc_ess
from blackjax.smc.resampling import systematic
import jax
jax.config.update("jax_enable_x64", True) #(LC)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from jimgw.core.prior import (
    CombinePrior,
    CosinePrior,
    SinePrior,
    UniformPrior,
)
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD #(LC)
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
    SkyFrameToDetectorFrameSkyPositionTransform,
)
from jimgw.core.single_event.transform_utils import eta_to_q   #(LC) different file name
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2   #new waveform (LC)

#updated for tidal waveform (LC)
PARAMETER_NAMES = [
    "M_c",
    "q",
    "s1_z",
    "s2_z",
    "lambda_1",
    "lambda_2",
    "d_L",
    "t_c",
    "phase_c",
    "iota",
    "psi",
    "ra",
    "dec",
]

M_c_min, M_c_max = 1.184, 2.168 #changed for GW170817
q_min, q_max = 0.125, 1.0

def build_prior() -> CombinePrior:
    """Prior matching the GW150914 setup."""

    prior = []

    #mass prior
    Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
    q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])
    prior.extend([Mc_prior, q_prior])

    #new spin prior with tidal parameters (LC)
    prior.extend(
        [
            UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
            UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
            UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
            UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
            SinePrior(parameter_names=['iota']),
        ]
    )
    # Extrinsic prior
    prior.extend(
        [
            UniformPrior(10.0, 75.0, parameter_names=["d_L"]),  #was power law before        
            UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),  
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
            UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
            CosinePrior(parameter_names=["dec"]),
        ]
    )

    return CombinePrior(prior)



#new transforms (LC)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nested-sampling GW170817 analysis with BlackJAX.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outdir/",
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--N",
        type=str,
        default="",
        help="Identifier appended to the GW170817 results directory.",
    )
    parser.add_argument(
        "--num-repeats",
        dest="num_repeats",
        type=int,
        default=1,
        help="Multiplier for inner HRSS steps (num_repeats * n_dims).",
    )
    return parser

def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None, overrides=None):
    args = parse_args(argv)
    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)

    base_outdir = args.outdir if args.outdir.endswith("/") else f"{args.outdir}/"
    tag = f"_{args.N}" if args.N else ""
    outdir = f"{base_outdir}GW170817{tag}/"   
    os.makedirs(outdir, exist_ok=True)

    print(f"Saving output to {outdir}")
    print("Starting data fetch and PSD estimation")
    print("getting data and waveform")

    #specific for GW170817
    gps = 1187008882.43  
    duration = 128.0   
    psd_duration = 2000
    psd_pad = 16
    start = gps + 2.0 - duration  
    end = start + duration   
    psd_start = start - psd_pad - psd_duration
    psd_end = start - psd_pad
    fmin = 20.0
    fmax = 1024  #goal is 2048

    ifos = [get_H1(), get_L1(), get_V1()]
    for ifo in ifos:
        data = Data.from_gwosc(ifo.name, start, end, version=2)
        ifo.set_data(data)

        psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end, version=2)
        psd_fftlength = data.duration * data.sampling_frequency
        ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))


    waveform=RippleIMRPhenomD_NRTidalv2(f_ref=fmin, use_lambda_tildes=False)   #new waveform (LC)

    print("building prior and transforms")
    prior = build_prior()
    sample_transforms, likelihood_transforms = build_transforms(gps, ifos)

    #Updated (LC)
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

    #removed previous work converting between q and eta as this is handled in likelihood_transforms
    ref_param = prepare_ref_params(ref_param, likelihood_transforms)

    print("building likelihood")

    #new likelihood (LC)
    likelihood = HeterodynedTransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        trigger_time=gps,
        f_min=fmin,
        f_max=fmax,
        n_bins=256,   #goal is 501
        prior=prior,
        reference_parameters=ref_param,
        #optimizer_popsize=10,
        #optimizer_n_steps=50,   #goal is 100
        likelihood_transforms=likelihood_transforms,
        phase_marginalization=True,
    )

    arg_snapshot = vars(args).copy()
    run_config = {
        "arguments": arg_snapshot,
        "settings": {
            "gps": gps,
            "fmin": fmin,
            "fmax": fmax,
        },
    }

    with open(os.path.join(outdir, "run_configuration.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)


    print("initialising transforms")
    def loglikelihood(x):
        for transform in reversed(sample_transforms):
            x, _ = transform.inverse(x)
        for transform in reversed(likelihood_transforms):
            x = transform.forward(x)
        like = likelihood.evaluate(x)   #(LC) had to remove 'none' due to new format
        return like

    def log_prior(x):
        transform_jacobian = 0.0
        for transform in reversed(sample_transforms):
            x, jacobian = transform.inverse(x)
            transform_jacobian += jacobian
        return prior.log_prob(x) + transform_jacobian

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
        return physical_samples


    n_dims = len(prior.parameter_names)
    n_live = 500   #goal is 5000
    n_delete = n_live // 2
    #num_mcmc_steps = args.num_repeats * n_dims    #goal is 8 x ndims
    num_mcmc_steps = 10  #reduced for quick runs

    labels = {
        "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
        "q": r"$q$",
        "s1_z": r"$s_{1,z}$",
        "s2_z": r"$s_{2,z}$",
        "lambda_1": r'$\Lambda_1$',
        "lambda_2": r'$\Lambda_2$',
        "iota": r"$\iota$",
        "d_L": r"$d_L\,[\mathrm{Mpc}]$",
        "t_c": r"$t_c\,[\mathrm{s}]$",
        "psi": r"$\psi$",
        "ra": r"$\alpha$",
        "dec": r"$\delta$",
    }

    print("initialising particles")
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key, 2)
    initial_particles = prior.sample(init_key, n_live)
    initial_particles = sample_to_unbound(initial_particles)

    first_key = next(iter(initial_particles.keys()))
    print(f"using device {initial_particles[first_key].device}")


    # ========== Run NSS ==========

    print("\n" + "=" * 50)
    print("Running Nested Sampling (NSS)")
    print("=" * 50)

    nested_sampler = blackjax.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=loglikelihood,
        num_delete=n_delete,
        num_inner_steps=num_mcmc_steps,
    )

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    #got rid of duplicate initialisation

    print("initialising sampler")
    state = nested_sampler.init(initial_particles)
    (state_dummy, rng_key), _ = one_step((state, rng_key), None)
    jax.block_until_ready(state_dummy.integrator.logZ)  #(LC) added .integrator

    dead = []
    with tqdm(desc="NSS Dead points", unit=" dead points") as pbar:
        while not state.integrator.logZ_live - state.integrator.logZ < -1:   #goal is -3, (LC) added 'integrator'
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    samples = blackjax.ns.utils.finalise(state, dead)
    nss_unbounded = jax.vmap(unbound_to_sample)(samples.particles.position)    #(LC) added .position
    nss_physical = process_samples_to_physical(nss_unbounded)
    nss_dataframe = NestedSamples(
        data=nss_physical,
        logL=samples.particles.loglikelihood,    #added .particles (LC)
        logL_birth=samples.particles.loglikelihood_birth,   #added same as above
        labels=labels,
    )


    nss_ess = float(blackjax.ns.utils.ess(jax.random.PRNGKey(0), samples))
    print(f"NSS log(Z) = {float(nss_dataframe.logZ()):.2f}")
    print(f"NSS ESS = {nss_ess:.1f}")

    nss_dataframe.to_csv(
        "./outdir/GW170817/nss_samples.csv"  
    )



if __name__ == "__main__":
    main()