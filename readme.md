Lab 1: Signal Analysis ToolboxDeveloped for UC Berkeley's Astro 121, this package provides a comprehensive suite of tools for digital signal processing, Fourier analysis, and mixer characterization. It is designed to process laboratory data sampled from sine wave generators and mixers.FeaturesSpectral Analysis: Perform FFTs, calculate power spectra, and analyze voltage amplitudes.Aliasing & Nyquist: Visualize Nyquist zones and simulate digital aliasing effects.Windowing & Leakage: Analyze spectral leakage using Flat-Top windows and high-resolution padded DFTs.Mixer Characterization: Tools for analyzing Double Sideband (DSB) and Single Sideband (SSB) mixer outputs, including IQ phase correction.Noise Statistics: Calculate RMS voltage, variance, and Gaussian fits for experimental noise data.InstallationTo install the package locally in editable mode (allowing you to modify the code and see changes immediately), run the following from the root directory:Bashpip install -e .
Dependencies:

The following libraries are required and will be installed automatically: 
    numpy: For numerical arrays and FFT calculations.
    matplotlib: For signal and spectral visualization.
    scipy: Specifically for curve_fit and flattop window functions.
    
Quick Start
1. Basic Sine Analysis

from lab1.Lab1 import sine_signal_analysis

# Analyze a 400kHz signal sampled at 2MHz
# signal_sample = (signal_khz, sample_mhz)
sine_signal_analysis(signal_sample=(400, 2.0), simulation=True)

2. Visualizing Aliasing

from lab1.Lab1 import nyquist_zones

# See how a high-frequency signal folds into the first Nyquist zone
nyquist_zones(signal_sample=(2500, 2.0))

3. SSB Mixer Analysis

from lab1.Lab1 import analyze_ssb_mixer, generate_simulated_ssb_data

# See a simulation of SSB data given inputs.
I, Q = generate_simulated_ssb_data(delta_nu_khz=10.0)
analyze_ssb_mixer(I, Q, fs_mhz=2.0, target_df_khz=10.0)


File Structure

read_sine_data: Helper to batch-load .npz laboratory data. Data must follow the filename test_{sample frequency}MHz_{signal frequency}kHz
files.spectral_resolution: Analyzes the ability to distinguish two nearby frequencies based on sample length $N$.
noise_SNR: Calculates the Signal-to-Noise Ratio scaling across multiple data blocks.
downsample: Artificially reduces the sample rate of a signal to test aliasing limits.

Important Global Variables
Note that several functions in this script expect the following constants to be defined in your environment:
IMPEDANCE: Defaulting to $50.0 \Omega$.
sine_data: A dictionary containing loaded lab samples.