def read_sine_data(folder, f_sample, f_signal):
    """
    Reads a list of data samples with the same path, but the indices for signal and sample rates change.

    Args:
        folder (string): file path for the folder including the data
        f_sample (list): all possible sample rates recorded in lab. In MHz (Hz*1e6).
        f_signal (list): all possible signal rates recorded in lab. In kHz (Hz*1e3).
    """
    
    data = {}
    
    for i in f_sample:
        for j in f_signal:
            filename = f"test_{i}MHz_{j}kHz.npz"
            filepath = os.path.join(base_dir, filename)
            data[f"{i}MHz {j}kHz"] = (np.load(filepath))["arr_0"][1:]
    return data

def sine_signal_analysis(signal_sample, simulation=False, block=1, 
                         sinewave=None, N=None, times=None):
    """
    Analyzes a sine wave signal. Produces plots of the sampled or a simulated wave, its fourier transform, power spectrum, and voltage spectrum.

    Args:
        signal_sample (tuple): Input signal frequency in kHz and sample frequency in MHz (signal, sample).
        simulation: 
            if False: reads and uses real data from the sine_data dictionary made earlier. 
            if True: creates a sine wave using the signal and sample rates inputted.
        block (int): which block of 2048 samples to use for data. Must be 1 through 5. Zeroth block can be used but has stale buffers.
        sinewave, N, times: used when a sample is inputted instead of being created by the function itself
            sinewave (np.array): raw data
            N (int): amount of samples in the data sample
            times (np.array): time domain of the data sample
    """
    # 1. Basic Setup
    sample = signal_sample[1]
    signal_freq = signal_sample[0]
    fs = sample * 1e6
    dt = 1 / fs

    # --- 2. Data Acquisition Logic ---
    # If no external data is provided, we must Load or Simulate it
    if sinewave is None:
        # Determine N (Length)
        if N is None:
            # Load specific block to check shape
            # Ensure 'sine_data' exists in your global scope!
            N = sine_data[f"{sample}MHz {signal_freq}kHz"][block].shape[0]
        
        # Generate Time Axis
        if times is None:
            times = np.arange(N) * dt
            
        # Get/Generate Amplitude & Signal
        # (Assuming you want to normalize to the max of the loaded block)
        raw_block = sine_data[f"{sample}MHz {signal_freq}kHz"][block] - np.mean(sine_data[f"{sample}MHz {signal_freq}kHz"][block])
        amp_sinewave = np.max(raw_block)

        if simulation == False:
            sinewave = raw_block
        else: # simulation == True
            # Generate pure sine wave
            sinewave = amp_sinewave * np.sin(2 * np.pi * (signal_freq * 1e3) * times)
            
    else:
        # External data was provided. 
        # We assume N and times are either provided or need to be derived.
        if N is None:
            N = len(sinewave)
        if times is None:
            times = np.arange(N) * dt
        # Calculate amp for reporting
        amp_sinewave = np.max(sinewave)

    # For plotting smooth continuous line (Simulation only)
    sinewave_sim = None
    if simulation:
        # High-res time axis for smooth curve
        t_high_res = np.linspace(0, times[-1], 50000)
        sinewave_sim = amp_sinewave * np.sin(2 * np.pi * (signal_freq * 1e3) * t_high_res)

    # --- 3. Calculations ---
    # A. Raw FFT (Unnormalized mathematical sum)
    fft_raw = np.fft.fft(sinewave)
    
    # B. Voltage Spectrum (Normalized by N to get Volts)
    fft_voltage = fft_raw / N
    
    # C. Power Spectrum (Watts)
    # P = |V|^2 / R
    # (Ensure IMPEDANCE is defined globally, e.g., 50.0)
    power_spectrum = (np.abs(fft_voltage)**2) / IMPEDANCE
    
    # Frequency Axis
    frequencies = np.fft.fftfreq(N, d=dt)
    
    # --- 4. Shift for Visualization ---
    # Shift everything so 0 Hz is in the center
    freqs_shifted = np.fft.fftshift(frequencies)
    
    raw_shifted = np.fft.fftshift(fft_raw)
    voltage_shifted = np.fft.fftshift(fft_voltage)
    power_shifted = np.fft.fftshift(power_spectrum)
    
    
    # --- 5. Validation Analysis ---
    # Find peak in positive frequencies
    pos_mask = frequencies > 0
    peak_idx = np.argmax(power_spectrum[pos_mask])
    # Map back to real frequencies
    detected_freq = frequencies[pos_mask][peak_idx]
    
    # Check complex values at the peak
    peak_complex = fft_voltage[pos_mask][peak_idx]
    # Handle phase wrap/calculation
    if peak_complex.real != 0:
        phase = np.arctan(peak_complex.imag / peak_complex.real)
    else:
        phase = 0
    
    # --- 6. Plotting ---
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), constrained_layout=True)
    
    # [Plot 1] Signal (Time Domain)
    # Plot only first 50 points or fewer if N is small
    plot_lim = min(50, N)
    axs[0,0].plot(times[:plot_lim]*1e6, sinewave[:plot_lim], color="purple", lw=2, label = "Sampled Wave")
    axs[0,0].scatter(times[:plot_lim]*1e6, sinewave[:plot_lim], color="purple", s=10)
    
    if simulation and sinewave_sim is not None:
        axs[0,0].plot(np.linspace(0, times[-1], 50000) * 1e6, sinewave_sim, color="purple", lw=2, alpha=0.5, label = "Continuous Wave")
        axs[0,0].legend(loc='upper right')
        
    axs[0,0].set_xlim(times[0]*1e6, times[min(35, N-1)]*1e6)
    axs[0,0].set_xlabel("Time (μs)")
    axs[0,0].set_ylabel("Voltage Amplitude (Arb.)")
    axs[0,0].set_title("1. Time Domain Signal")
    axs[0,0].grid(True, alpha=0.3)
    
    # [Plot 2] Raw FFT (Real & Imaginary)
    axs[0,1].plot(freqs_shifted/1e3, raw_shifted.real, lw=2, label="Real")
    axs[0,1].plot(freqs_shifted/1e3, raw_shifted.imag, lw=2, label="Imaginary", color='tab:red', alpha=0.7)
    axs[0,1].set_xlim(-detected_freq/1e3-50, detected_freq/1e3+50)
    axs[0,1].legend()
    axs[0,1].set_xlabel("Frequency (kHz)")
    axs[0,1].set_ylabel("Raw Amplitude (Counts)")
    axs[0,1].set_title("2. Raw FFT")
    axs[0,1].grid(True, alpha=0.3)
    
    # [Plot 3] Power Spectrum
    axs[1,0].semilogy(freqs_shifted/1e3, power_shifted, lw=2)
    axs[1,0].set_xlim(-detected_freq/1e3-50, detected_freq/1e3+50)
    axs[1,0].set_xlabel("Frequency (kHz)")
    axs[1,0].set_ylabel("Power (Arb.)")
    axs[1,0].set_title("3. Power Spectrum")
    axs[1,0].grid(True, alpha=0.3)
    
    # [Plot 4] Voltage Spectrum (Real & Imaginary)
    axs[1,1].plot(freqs_shifted/1e3, voltage_shifted.real, lw=2, label="Real", color='tab:blue')
    axs[1,1].plot(freqs_shifted/1e3, voltage_shifted.imag, lw=2, label="Imaginary", color='tab:red', alpha=0.7)
    axs[1,1].set_xlim(-detected_freq/1e3-50, detected_freq/1e3+50)
    axs[1,1].legend()
    axs[1,1].set_xlabel("Frequency (kHz)")
    axs[1,1].set_ylabel("Voltage Amplitude (Arb.)")
    axs[1,1].set_title("4. Voltage Spectrum")
    axs[1,1].grid(True, alpha=0.3)
    
    title_prefix = "Simulated Sine Wave" if simulation else "Sampled Sine Wave"
    fig.suptitle(f"{title_prefix} Analysis: {sample} MHz Sample Rate, {signal_freq} kHz Signal", fontsize=16)
    plt.show()
    
    print(f"--- Numerical Results ---")
    print(f"Target Freq:    {signal_freq} kHz")
    print(f"Detected Freq:  {detected_freq/1e3:.4f} kHz")
    print(f"Amplitude: {amp_sinewave:.4f} Arb. Units")
    print(f"Phase: {phase/np.pi:.3f}π rad")
    
    return sinewave

def IFFT(sinewave, signal_sample, block=1):
    """
    This function performs an inverse fourier transform on an inputted sine wave. It pads zeroes to the beginning and end to
    accurately depict its long-period shape of having a negative linear dependency to time.

    Args:
        sinewave (np.array): inputted signal
        signal_sample (tuple): Input signal frequency in kHz and sample frequency in MHz (signal, sample).
        block (int): which block of 2048 samples to use for data. Must be 1 through 5. Zeroth block can be used but has stale buffers.
    """

    
    sample = signal_sample[1]
    signal = signal_sample[0]
    fs = sample * 1e6
    dt = 1 / fs
    N = sine_data[f"{sample}MHz {signal}kHz"][block].shape[0]

    N_pad = 2*N
    fft_padded = np.pad(sinewave, (0, N), 'constant')
    power_spectrum_padded = np.abs(np.fft.fft(fft_padded))**2
    ifft_padded = np.fft.fftshift(np.fft.ifft(power_spectrum_padded).real)
    lags_padded = np.arange((-N_pad//2), (N_pad//2)) * dt
    
    fig, axs = plt.subplots(ncols=2, figsize = (15, 5), sharey=True, gridspec_kw={'width_ratios': [1, 3]})
    axs[0].plot(lags_padded*1e6, ifft_padded / np.max(ifft_padded), color="green", lw=2)
    axs[0].scatter(lags_padded*1e6, ifft_padded / np.max(ifft_padded), color="green", s=7)
    axs[0].set_xlim(lags_padded[N]*1e6, lags_padded[N+50]*1e6)
    axs[0].set_xlabel("time (μs)")
    axs[0].set_ylabel("Voltage (V)")
    axs[0].set_title("IFFT of Sine Wave: Short-Period Trends")
    axs[1].plot(lags_padded*1e3, ifft_padded / np.max(ifft_padded), color = "green")
    axs[1].set_xlabel("time (ms)")
    axs[1].set_title("IFFT of Sine Wave: Long-Period Trends")
    axs[1].set_xlim(min(lags_padded)*1e3, max(lags_padded)*1e3)
    fig.suptitle("Inverse FFT on Sampled Sine Wave", fontsize = 20)
    plt.tight_layout()
    plt.show()

def correlate(sinewave, signal_sample, block=1):
    sample = signal_sample[1] 
    dt = 1 / (sample * 1e6)
    
    # Extract the raw signal array (assuming tuple input like IFFT)
    N = sinewave.shape[0]

    # 2. Perform Time-Domain Autocorrelation
    # mode='full' returns the convolution of the signal with itself inverted.
    # For a real signal, this is the linear autocorrelation.
    # Result length will be 2*N - 1
    autocorr = np.correlate(sinewave, sinewave, mode='full')
    # 3. Define Lags
    # The 'full' mode places lag 0 at index (N-1)
    # Lags range from -(N-1) to +(N-1)
    lags = np.arange(-(N-1), N) * dt
    
    # 4. Normalize
    # Normalize by the maximum value (Lag 0) to match the IFFT plot scaling
    autocorr_norm = autocorr / np.max(autocorr)
    
    # 5. Plotting (Mimicking the IFFT function exactly)
    fig, axs = plt.subplots(ncols=2, figsize=(15, 5), sharey=True, gridspec_kw={'width_ratios': [1, 3]})
    
    # --- Plot 1: Short-Period Trends (Zoomed in on Center) ---
    axs[0].plot(lags*1e6, autocorr_norm, lw=2, color="red")
    axs[0].scatter(lags*1e6, autocorr_norm, s=7, color="red")
    
    # Zoom logic: Center is index N-1. We show Center to Center+50 samples
    center_idx = N - 1
    axs[0].set_xlim(lags[center_idx]*1e6, lags[center_idx + 50]*1e6)
    
    axs[0].set_xlabel("time (μs)") # Note: Your original said µs but plotted seconds. I kept standard 's' to be safe.
    axs[0].set_ylabel("Voltage (Arb.)") # Normalized correlation is unitless (0-1), but keeping your label.
    axs[0].set_title("Direct Correlation of Sine Wave: Short-Period Trends")
    
    # --- Plot 2: Long-Period Trends (Full View) ---
    axs[1].plot(lags*1e3, autocorr_norm, color="red")
    axs[1].set_xlabel("time (ms)")
    axs[1].set_title("Direct Correlation of Sine Wave: Long-Period Trends")
    axs[1].set_xlim(min(lags)*1e3, max(lags)*1e3)
    
    fig.suptitle("Time-Domain Autocorrelation of Sampled Sine Wave", fontsize = 20)
    plt.tight_layout()
    plt.show()

def nyquist_zones(signal_sample, downsample = 1):
    sample = signal_sample[1]*1e6 / downsample
    signal = signal_sample[0]*1e3
    nyquist_freq = sample/2

    print(f"Nyquist Frequency: {nyquist_freq/1e3}kHz")
    print(f"Signal Frequency: {sample/1e3}kHz")
    
    # 1. Calculate Baseband Alias (Zone 1 equivalent)
    # This uses the absolute distance to the nearest integer multiple of fs
    f_alias = np.abs(signal - sample * np.round(signal / sample))
    
    # 2. Determine number of zones to plot (minimum 2)
    needed_zones = int(np.ceil(signal / nyquist_freq))
    num_zones = max(2, needed_zones)
    
    plt.figure(figsize=(12, 4))
    
    # Loop to draw Nyquist Windows
    for i in range(num_zones):
        z_start, z_end = i * nyquist_freq, (i + 1) * nyquist_freq
        z_num = i + 1
        
        # Color odd zones red (per your request)
        if z_num % 2 != 0:
            plt.axvspan(z_start/1e3, z_end/1e3, color="red", alpha=0.3, label="Odd Zone (Normal)" if i==0 else "")
        else:
            plt.axvspan(z_start/1e3, z_end/1e3, color="gray", alpha=0.1, label="Even Zone (Reversed)" if i==1 else "")
        
        plt.text((z_start + z_end)/2e3, 0.85, f"Zone {z_num}", ha='center', weight='bold')
    
    # --- Plot the Frequencies ---
    if signal == f_alias:
        plt.axvline(signal/1e3, color="green", linewidth=3, label=f"Signal ({signal/1e3} kHz)")
    else:
        plt.axvline(signal/1e3, color="green", linewidth=3, label=f"Actual Signal ({signal/1e3} kHz)")
        plt.axvline(f_alias/1e3, color="blue", linewidth=3, label=f"Aliased Signal ({f_alias/1e3} kHz)")
    
    
    # --- Formatting ---
    plt.axvline(sample/1e3, color="black", linewidth=2, label=f"Sampling Freq {sample/1e6}MHz")
    plt.axvline(nyquist_freq/1e3, color="red", linestyle="--", label=f"Nyquist Freq {sample/2e6}MHz")
    
    plt.xlim(0, (num_zones * nyquist_freq)/1e3)
    plt.ylim(0, 1)
    plt.title(f"Visualizing Aliasing: Signal at {signal/1e3}kHz folds to {f_alias/1e3}kHz")
    plt.xlabel("Frequency (kHz)")
    plt.yticks([])
    plt.legend(loc='upper right', framealpha=1)
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    
    plt.show()

def aliasing(signal_sample, downsample=1, N=2048):
    sample = signal_sample[1]*1e6/downsample
    signal = signal_sample[0]*1e3
    nyquist_freq = sample/2
    N = N//downsample
    f_alias = np.abs(signal - sample * np.round(signal / sample))
    
    print(f"--- Configuration ---")
    print(f"Effective Sample Rate: {sample/1e3:.2f} kHz")
    print(f"Nyquist Limit:         {nyquist_freq/1e3:.2f} kHz")
    print(f"Input Signal:          {signal/1e3:.2f} kHz")
    print(f"Expected Appearance:   {f_alias/1e3:.2f} kHz (Zone 1)")
    
    # --- 3. Time Domain Setup ---
    # We determine the plot duration based on the ALIAS frequency.
    # We want to see ~5 cycles of the resulting wave to understand what the digital system 'sees'.
    cycles_to_show = 5
    if f_alias > 0:
        duration = cycles_to_show / f_alias
    else:
        # If alias is exactly 0 (DC) or extremely close, default to showing 5 cycles of fs
        duration = cycles_to_show / (sample/10)
    
    # High-Res Analog Time (100x faster than signal or sample rate to look smooth)
    # We ensure it's fine enough to draw the high-freq input signal
    max_freq = max(sample, signal)
    t_analog = np.arange(0, duration, 1/(100 * max_freq))
    
    # Digital Sample Time (Exact sample spacing)
    t_sample = np.arange(0, duration, 1/sample)
    
    # --- 4. Signal Generation ---
    # The "Truth" (Analog high-freq signal)
    analog_signal = np.sin(2 * np.pi * signal * t_analog)
    
    # The "Measurement" (Sampled points)
    sampled_signal = np.sin(2 * np.pi * signal * t_sample)
    
    # The "Illusion" (The sine wave the samples *suggest* exists in Zone 1)
    # Note: Phase might need adjustment depending on folding, but freq is correct
    alias_wave_guide = np.sin(2 * np.pi * f_alias * t_analog) 
    # If in an even zone, phase effectively flips (sign inversion often occurs)
    zone_idx = int(round(signal / nyquist_freq))
    # Simple check: if samples correlate negatively with the positive alias guide, flip the guide for visual
    if np.dot(np.interp(t_sample, t_analog, alias_wave_guide), sampled_signal) < 0:
        alias_wave_guide *= -1
    
    # --- 5. Frequency Domain (FFT) ---
    # Use a separate longer signal for FFT to get clean peaks (independent of plot duration)
    t_fft = np.arange(N) * (1/sample)
    signal_for_fft = np.sin(2 * np.pi * signal * t_fft)
    fft_vals = np.fft.fft(signal_for_fft)
    fft_freqs = np.fft.fftfreq(N, d=1/sample)
    
    # Normalize Magnitude
    fft_mag = np.abs(fft_vals) / N 
    
    # --- 6. Visualization ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    
    # [Plot 1] The Analog Reality
    axs[0].plot(t_analog * 1e6, analog_signal, color='green', alpha=0.5, label=f"True Signal ({signal/1e3:.1f} kHz)")
    axs[0].set_title(f"1. Input Signal")
    axs[0].set_ylabel("Amplitude (Arb.)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_xlim(0, duration/2*1e6)
    
    # [Plot 2] The Sampling Process (Strobe Effect)
    # Show Analog ghost
    axs[1].plot(t_analog * 1e6, analog_signal, color='green', alpha=0.15, label="True Signal")
    axs[1].stem(t_sample * 1e6, sampled_signal, linefmt='b-', markerfmt='bo', basefmt=" ", label=f"Samples (@ {sample/1e3:.1f} kHz)")
    axs[1].plot(t_analog * 1e6, alias_wave_guide, color='blue', linestyle='--', alpha=0.4, label=f"Aliased Signal ({f_alias/1e3:.1f} kHz)")
    
    axs[1].set_title(f"2. Sampled and Aliased Signal")
    axs[1].set_ylabel("Amplitude (Arb.)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)
    axs[1].set_xlim(0, duration/2*1e6)
    
    # [Plot 3] The FFT Spectrum (1st Nyquist Zone)
    axs[2].plot(np.fft.fftshift(fft_freqs/1e3), np.fft.fftshift(fft_mag), color='red', lw=2)
    axs[2].set_title("3. Fourier Transform of Signal (found in the first Nyquist Zone)")
    axs[2].set_xlabel("Frequency (kHz)")
    axs[2].set_ylabel("Magnitude (Normalized)")
    axs[2].set_xlim(0, nyquist_freq/1e3) 
    
    # Highlight the Alias Peak
    axs[2].axvline(f_alias/1e3, color='blue', linestyle='--', alpha=0.6, label=f"Detected Peak @ {f_alias/1e3:.1f} kHz")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.show()

def downsample(signal_sample, sinewave, downsample):
    """
    Creates a downsampled signal using an input sine wave, signal frequency, sample rate, and downsampling fraction.
    
    Args:
        signal_sample (tuple): Input signal frequency in kHz and sample frequency in MHz (signal, sample).
        sinewave (np.array): The time-domain signal array (volts)
        downsample (int, power of 2): System impedance in Ohms (default 50.0)
    """
    sample = signal_sample[1]
    fs = sample * 1e6
    dt = 1 / fs
    N = sinewave.shape[0]
    times = np.arange(N) * dt

    downsample_idx = []
    for i in range(N):
        if i%downsample != 0:
            downsample_idx.append(i)
    sinewave_downsampled = np.delete(sinewave, downsample_idx)
    times_downsampled = np.delete(times, downsample_idx)
    return sinewave_downsampled, times_downsampled

def downsampled_signal(sinewave, signal_sample, downsample = 1):
    """
    Performs FFT analysis on a signal, plots Time/Raw/Power/Voltage domains, 
    and validates the detected frequency against the target.
    
    Args:
        signal_sample (tuple): Input signal frequency in kHz and sample frequency in MHz (signal, sample).
        target_freq_khz (float): Expected signal frequency (for aliasing check)
        impedance (float): System impedance in Ohms (default 50.0)
    """
    signal = signal_sample[0]*1e3
    sample = signal_sample[1]*1e6 / downsample
    # --- 1. Basic Parameters ---
    N = sinewave.shape[0]
    dt = 1 / sample
    times = np.arange(N) * dt
    
    # --- 2. Calculations ---
    # A. Raw FFT (Unnormalized mathematical sum)
    fft_raw = np.fft.fft(sinewave)
    
    # B. Voltage Spectrum (Normalized by N to get Volts)
    fft_voltage = fft_raw / N
    
    # C. Power Spectrum (Watts)
    # P = |V|^2 / R
    power_spectrum = (np.abs(fft_voltage)**2) / IMPEDANCE
    
    # Frequency Axis
    frequencies = np.fft.fftfreq(N, d=dt)
    
    # --- 3. Shift for Visualization ---
    # Shift everything so 0 Hz is in the center
    freqs_shifted = np.fft.fftshift(frequencies)
    
    raw_shifted = np.fft.fftshift(fft_raw)
    voltage_shifted = np.fft.fftshift(fft_voltage)
    power_shifted = np.fft.fftshift(power_spectrum)
    
    # --- 4. Validation Analysis (Calculated before plotting for title usage) ---
    # Find peak in positive frequencies
    pos_mask = frequencies > 0
    peak_idx = np.argmax(power_spectrum[pos_mask])
    
    # Map back to real frequencies
    detected_freq_hz = frequencies[pos_mask][peak_idx]
    detected_freq_khz = detected_freq_hz / 1e3
    
    # Check complex values at the peak
    peak_complex = fft_voltage[pos_mask][peak_idx]
    if peak_complex.real != 0:
        phase = np.arctan(peak_complex.imag / peak_complex.real)
    else:
        phase = 0
        
    # --- 6. Plotting ---
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), constrained_layout=True)
    
    # [Plot 1] Signal (Time Domain)
    # Plot only first 50 points or fewer if N is small
    plot_lim = min(50, N)
    axs[0,0].plot(times[:plot_lim]*1e6, sinewave[:plot_lim], color="purple", lw=2, label = "Sampled Wave")
    axs[0,0].scatter(times[:plot_lim]*1e6, sinewave[:plot_lim], color="purple", s=10)
    axs[0,0].set_xlim(times[0]*1e6, times[min(35, N-1)]*1e6)
    axs[0,0].set_xlabel("Time (μs)")
    axs[0,0].set_ylabel("Voltage Amplitude (Arb.)")
    axs[0,0].set_title("1. Time Domain Signal")
    axs[0,0].grid(True, alpha=0.3)
    
    # [Plot 2] Raw FFT (Real & Imaginary)
    axs[0,1].plot(freqs_shifted/1e3, raw_shifted.real, lw=2, label="Real")
    axs[0,1].plot(freqs_shifted/1e3, raw_shifted.imag, lw=2, label="Imaginary", color='tab:red', alpha=0.7)
    axs[0,1].set_xlim(-detected_freq_khz/1e3-50, detected_freq_khz/1e3+50)
    axs[0,1].legend()
    axs[0,1].set_xlabel("Frequency (kHz)")
    axs[0,1].set_ylabel("Raw Amplitude (Sum)")
    axs[0,1].set_title("2. Raw FFT (Unnormalized)")
    axs[0,1].grid(True, alpha=0.3)
    
    # [Plot 3] Power Spectrum
    axs[1,0].semilogy(freqs_shifted/1e3, power_shifted, lw=2)
    axs[1,0].set_xlim(-detected_freq_khz/1e3-50, detected_freq_khz/1e3+50)
    axs[1,0].set_xlabel("Frequency (kHz)")
    axs[1,0].set_ylabel("Power (Watts)")
    axs[1,0].set_title("3. Power Spectrum")
    axs[1,0].grid(True, alpha=0.3)
    
    # [Plot 4] Voltage Spectrum (Real & Imaginary)
    axs[1,1].plot(freqs_shifted/1e3, voltage_shifted.real, lw=2, label="Real", color='tab:blue')
    axs[1,1].plot(freqs_shifted/1e3, voltage_shifted.imag, lw=2, label="Imaginary", color='tab:red', alpha=0.7)
    axs[1,1].set_xlim(-detected_freq_khz/1e3-50, detected_freq_khz/1e3+50)
    axs[1,1].legend()
    axs[1,1].set_xlabel("Frequency (kHz)")
    axs[1,1].set_ylabel("Voltage Amplitude (V)")
    axs[1,1].set_title("4. Voltage Spectrum (Normalized)")
    axs[1,1].grid(True, alpha=0.3)
    
    fig.suptitle(f"Downsampled Signal Analysis: {sample/1e3} kHz Sample Rate, {signal/1e3} kHz Signal", fontsize=16)
    plt.show()
    
    # --- 6. Print Results ---
    print(f"--- Analysis Results ---")
    print(f"Target Freq:    {signal/1e3} kHz")
    print(f"Detected Freq:  {detected_freq_khz:.4f} kHz")
    print(f"Phase: {phase/np.pi:.3f}π rad")
    
    # Aliasing Check (1% tolerance)
    error = np.abs((signal - detected_freq_khz) / signal)
    if error > 0.01:
        print("Aliasing? Yes (Detected frequency does not match Target)")
    else:
        print("Aliasing? No")

def get_flat_top_peak(signal):
    """
    Applies Flat-Top Windowing to a single signal array 
    and returns the peak voltage.
    """
    N = len(signal)
    
    # 1. Create Window & Correction Factor
    w_flat = flattop(N)
    acf = 1 / np.mean(w_flat) # Amplitude Correction Factor

    # 2. Apply Window
    sig_windowed = signal * w_flat
    
    # 3. FFT
    fft_val = np.fft.rfft(sig_windowed)
    
    # 4. Magnitude (Volts)
    # Scale: |FFT| / N * 2 (single-sided) * ACF
    magnitude = (np.abs(fft_val) / N) * 2 * acf
    
    # 5. Return Max
    return np.max(magnitude)

def spectral_leakage(sinewave, signal_sample, pad_factor = 100):
    # --- 1. Configuration & Variables ---
    sample = signal_sample[1]
    signal = signal_sample[0]
    
    # Convert to base units
    dt = 1 / (sample * 1e6)
    N = sinewave.shape[0]
    
    print(f"--- Analysis Parameters ---")
    print(f"Sample Count (N):    {N}")
    print(f"Sample Rate:         {sample:.1f} MHz")
    print(f"Signal Frequency:    {signal:.1f} kHz")
    
    # --- 3. Lab Requirement: 5.5 Leakage Power Analysis ---
    
    # A. Standard DFT (Nfreq = N)
    # "The dft by default calculates a Fourier spectrum at N frequencies..."
    # "FFT operations hard-code this spacing."
    fft_std = np.fft.fft(sinewave)
    freqs_std = np.fft.fftfreq(N, d=dt)
    
    # Calculate Power (Watts) for Standard DFT
    # Voltage V = |FFT| / N
    # Power P = V^2 / R
    V_std = np.abs(fft_std) / N
    P_std = (V_std**2) / IMPEDANCE
    P_db_std = 10 * np.log10(P_std + 1e-15)
    
    # Shift for visualization
    freqs_std_shift = np.fft.fftshift(freqs_std)
    P_db_std_shift = np.fft.fftshift(P_db_std)
    
    
    # B. High-Resolution DFT (Nfreq >> N)
    # "Calculate the power spectrum for waveforms where Nfreq >> N."
    # "Use dft with delta_v << vs/N."
    N_freq = N * pad_factor
    
    fft_pad = np.fft.fft(sinewave, n=N_freq)
    freqs_pad = np.fft.fftfreq(N_freq, d=dt)
    
    # Calculate Power for Padded DFT
    # CRITICAL: Normalize by N (the actual energy source), NOT N_freq
    V_pad = np.abs(fft_pad) / N
    P_pad = (V_pad**2) / IMPEDANCE
    P_db_pad = 10 * np.log10(P_pad + 1e-15)
    
    # Shift for visualization
    freqs_pad_shift = np.fft.fftshift(freqs_pad)
    P_db_pad_shift = np.fft.fftshift(P_db_pad)
    
    
    # --- 4. Visualization (Convolution Theorem) ---
    plt.figure(figsize=(12, 7))
    
    # Plot 1: The "True" continuous-like spectrum (Sinc Function)
    # This reveals the spectral leakage (lobes)
    plt.plot(freqs_pad_shift/1e3, P_db_pad_shift, color='tab:orange', linewidth=2,
             label=f'High-Res Spectrum ($N_{{freq}}={N_freq}$)\nShows Sinc Skirts (Leakage)')
    
    # Plot 2: The Standard DFT (Discrete Samples)
    # This shows why standard DFT might miss the exact peak or show "smeared" energy
    plt.plot(freqs_std_shift/1e3, P_db_std_shift, 'bo', markersize=2,
             label=f'Standard DFT ($N={N}$)\nHard-coded Spacing')
    plt.plot(freqs_std_shift/1e3, P_db_std_shift,
             label=f'Standard DFT ($N={N}$)\nHard-coded Spacing')
    
    # Formatting
    plt.title(f"Leakage Power: {signal} kHz Signal sampled at {sample} MHz\n"
              f"(Visualizing the Convolution Theorem)", fontsize=14)
    plt.xlabel("Frequency (kHz)", fontsize=12)
    plt.ylabel("Power (dBW)", fontsize=12)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='upper right')
    
    # Axis limits to focus on the 400kHz Signal
    plt.xlim(signal - 10, signal + 10) # Zoom: +/- 50kHz around signal
    plt.ylim(np.max(P_db_pad_shift) - 80, np.max(P_db_pad_shift) + 10) # Show top 80dB
    
    plt.show()

def spectral_resolution(signal_full, fs, N_values, signal_freqs):
    """
    Slices the signal to different N lengths and plots the spectrum 
    in both Logarithmic (dBW) and Linear (Watts) scales.
    """
    f1 = signal_freqs[0]
    f2 = signal_freqs[1]
    # Create a grid: Rows = number of N cases, Cols = 2 (Log vs Linear)
    fig, axes = plt.subplots(len(N_values), 2, figsize=(16, 5 * len(N_values)))
    
    pad_factor = 100 
    
    # Handle the case where N_values has only one element (axes would be 1D)
    if len(N_values) == 1:
        axes = np.expand_dims(axes, axis=0)
    plt.suptitle(f"Identifying Spectral Resolution of Mixed Signal: {f1}kHz and {f2}kHz", fontsize=20, y=1)
    for i, n_slice in enumerate(N_values):
        # 1. Processing
        sig_slice = signal_full[:n_slice]
        N_freq = n_slice * pad_factor
        fft_pad = np.fft.fft(sig_slice, n=N_freq)
        freqs = np.fft.fftfreq(N_freq, d=1/fs)
        
        # Power Calculation
        V = np.abs(fft_pad) / n_slice
        P_linear = (V**2) / IMPEDANCE
        P_db = 10 * np.log10(P_linear + 1e-15)
        
        # Shift
        freqs_shift = np.fft.fftshift(freqs)
        P_db_shift = np.fft.fftshift(P_db)
        P_lin_shift = np.fft.fftshift(P_linear)
        
        
        # Theoretical Resolution
        res_bw = fs / n_slice
        status = "RESOLVED" if (abs(f2-f1)*1e3) > res_bw else "UNRESOLVED"
        color = 'green' if status == "RESOLVED" else 'red'

        # --- Plot A: Logarithmic (dBW) ---
        ax_log = axes[i, 0]
        ax_log.plot(freqs_shift/1e3, P_db_shift, color='tab:orange', lw=1.5)
        ax_log.set_title(f"N={n_slice} | Log Scale (dBW)\nRes: {res_bw/1e3:.2f} kHz | {status}", 
                         color=color)
        ax_log.set_ylabel("Power (dBW)")
        
        # --- Plot B: Linear (Watts) ---
        ax_lin = axes[i, 1]
        ax_lin.plot(freqs_shift/1e3, P_lin_shift, color='tab:blue', lw=1.5)
        ax_lin.set_title(f"N={n_slice} | Linear Scale (Watts)\nRes: {res_bw/1e3:.2f} kHz | {status}", 
                         color=color)
        ax_lin.set_ylabel("Power (W)")

        # Formatting for both columns
        for ax in [ax_log, ax_lin]:
            ax.axvline(f1, color='black', linestyle='--', alpha=0.4, label='Input Frequencies')
            ax.axvline(f2, color='black', linestyle='--', alpha=0.4)
            ax.set_xlim((abs(freqs_shift[np.argmax(P_lin_shift)]) - 15e3)/1e3, (abs(freqs_shift[np.argmax(P_lin_shift)]) + 15e3)/1e3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Frequency (kHz)")
            ax.legend()
    plt.tight_layout()
    plt.show()

def noise_stats(full_noise_set):
    mu = np.mean(full_noise_set)
    variance = np.var(full_noise_set)
    sigma = np.sqrt(variance) # RMS Voltage
    
    print(f"Mean Voltage: {mu:.4f}")
    print(f"Variance:     {variance:.4f}")
    print(f"RMS (Sigma):  {sigma:.4f}")
    
    plt.figure(figsize=(10, 6))
    
    # Plot Histogram (Normalized to Density)
    count, bins, ignored = plt.hist(full_noise_set, bins=(max(full_noise_set)-min(full_noise_set)), density=True, 
                                    alpha=0.6, color='tab:blue', label='Measured Histogram')
    
    # Plot Theoretical Gaussian
    # Formula: G(x) = 1/(sigma*sqrt(2pi)) * exp(-0.5 * ((x-mu)/sigma)^2)
    y_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bins - mu) / sigma)**2)
    y_gauss_actual = (1 / (9.652 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bins - mu) / 9.652)**2)
    
    plt.plot(bins, y_gauss, linewidth=3, color='tab:orange', label=f'Theoretical Gaussian\n($\sigma$={sigma:.3f})')
    plt.plot(bins, y_gauss_actual, linewidth=3, color='tab:green', label=f'Experimental Gaussian\n($\sigma$={9.652:.3f})')
    plt.title("Noise Histogram")
    plt.xlabel("Voltage (Arb.)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def noise_SNR(noise_data, signal_sample, averages_to_plot):

    NUM_BLOCKS, BLOCK_SIZE = noise_data.shape
    TOTAL_SAMPLES = NUM_BLOCKS * BLOCK_SIZE
    FS = signal_sample[1]

    freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1/FS)
    all_psds = []
    
    # Calculate PSD for each block individually
    for i in range(noise_data.shape[0]):
        block = noise_data[i, :] # Get the i-th row
        fft_val = np.fft.rfft(block)
        psd = (np.abs(fft_val)**2) / BLOCK_SIZE
        all_psds.append(psd)
    
    all_psds = np.array(all_psds) # Shape (16, 1025)
    
    # Compare averages of N blocks
    plt.figure(figsize=(12, 12))
    SNR = []
    for i, N_avg in enumerate(averages_to_plot):
        if N_avg > NUM_BLOCKS: break 
        
        # Average the first N_avg blocks
        avg_psd = np.mean(all_psds[:N_avg], axis=0)
        avg_psd_db = 10 * np.log10(avg_psd + 1e-15)
    
        avg_psd_1 = np.mean(all_psds[:1], axis=0)
        avg_psd_db_1 = 10 * np.log10(avg_psd_1 + 1e-15)
        
        # Estimate SNR (Signal / Noise of the spectrum itself)
        # We define SNR here as Mean / StdDev (inverse of coefficient of variation)
        # We ignore DC and low freq garbage for the calculation
        valid_region = avg_psd[10:] 
        snr_est = np.mean(valid_region) / np.std(valid_region)
        SNR.append(snr_est)
        
        plt.subplot(len(averages_to_plot), 1, i+1)
        if i == 0:
            plt.plot(freqs/1e3, avg_psd_db_1, color='tab:green', lw=2,label=f"Spectrum for 1 block")
        else:
            plt.plot(freqs/1e3, avg_psd_db_1, color='black', lw=2,label=f"Spectrum for 1 block")
            plt.plot(freqs/1e3, avg_psd_db, color='tab:green', lw=2, label=f"Spectrum for {2**i} blocks")
        plt.title(f"Average of {N_avg} Blocks (Spectral SNR $\\approx$ {snr_est:.2f})")
        plt.legend(loc="upper right")
        plt.ylabel("Power (dB)")
        plt.grid(True, alpha=0.3)
        if i == len(averages_to_plot)-1: plt.xlabel("Frequency (kHz)")
    
    plt.tight_layout()
    plt.show()

    return SNR

def power_model(N, x):
    return N**x

def noise_scaling(N, SNR):
    # --- 1. Data Setup ---
    y_data = np.array(SNR)
    
    # --- 3. Perform the Fit ---
    # p0 is an initial guess for x
    popt, pcov = curve_fit(power_model, N, SNR, p0=[0.5])
    x_fitted = popt[0]
    perr = np.sqrt(np.diag(pcov))[0] # Standard deviation of the fit
    
    # --- 4. Visualization ---
    N_smooth = np.linspace(1, 16, 100)
    y_fitted = power_model(N_smooth, x_fitted)
    
    # --- 5. Results ---
    print(f"Fitted Exponent (x): {x_fitted:.4f} ± {perr:.4f}")
    return x_fitted, perr

def get_fwhm(x_axis, y_values):
    """Helper to calculate FWHM for a centered peak."""
    # Find the peak value and its location
    half_max = np.max(abs(y_values)) / 2.0 + 1e-16
    # Find indices where signal is above half-max
    indices = np.where(y_values >= half_max)[0]
    if len(indices) < 2:
        return 0
    # Width = difference between the first and last index on the x-axis
    return x_axis[indices[-1]] - x_axis[indices[0]]

def noise_ACF_PS(noise_data, signal_sample):
    """
    Compares Lab Noise and Lab Sine Data.
    Plots their Power Spectra and their Autocorrelations side-by-side.
    """
    fs = signal_sample[1]
    # 1. Setup
    N = len(noise_data)
    dt = 1 / fs
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    lags = np.arange(-(N-1), N) * dt
    
    # 2. Process Noise
    noise_centered = noise_data - np.mean(noise_data)
    noise_fft = np.fft.fftshift(np.fft.fft(noise_centered))
    noise_psd = (np.abs(noise_fft) / N)**2 / IMPEDANCE
    # Autocorrelation
    corr_noise = np.correlate(noise_centered, noise_centered, mode='full')
    corr_noise /= np.max(corr_noise) # Normalize

    fwhm_psd = get_fwhm(freqs, noise_psd)
    fwhm_acf = get_fwhm(lags, corr_noise)
    # 4. Plotting
    fig, axs = plt.subplots(nrows=2, figsize=(6, 6))

    # --- BOTTOM ROW: AUTOCORRELATIONS ---
    # Noise Autocorr (Zoomed to see the spike structure)
    axs[1].plot(lags * 1e6, corr_noise, color='tab:blue')
    axs[1].set_xlim(-0.2e10, 0.2e10) 
    axs[1].set_title("Autocorrelation of Noise")
    axs[1].set_xlabel("Lag (μs)")
    axs[1].set_ylabel("Normalized Correlation (Arb.)")
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)

    # --- TOP ROW: POWER SPECTRA ---
    # Noise PSD
    axs[0].plot(freqs, noise_psd, color='tab:green')
    axs[0].set_xlim(-0.2, 0.2)
    axs[0].set_title("Power Spectrum of Noise")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Relative Power (Arb.)")

    plt.tight_layout()
    plt.show()

    print(f"PSD FWHM: {fwhm_psd:.2f} Hz")
    print(f"ACF FWHM: {fwhm_acf * 1e6:.4f} μs")
    print(f"Product (FWHM_psd * FWHM_acf): {fwhm_psd * fwhm_acf:.4f}")

def analyze_dsb_data(data_array, nu_lo_khz, delta_nu_khz, fs_mhz, sideband_name="Upper", harmonics="False"):
    """
    Analyzes provided DSB mixer data, performs spectral analysis, and filters out the sum frequency.
    
    Args:
        data_array (np.array): The 2048-sample data array (DSB_upper or DSB_lower)
        nu_lo_khz (float): Local Oscillator frequency used during the experiment
        delta_nu_khz (float): The frequency offset (nu_RF - nu_LO)
        fs_mhz (float): The sample rate used by the digitizer
        sideband_name (str): Label for the plot ("Upper" or "Lower")
    """
    # --- 1. Constants ---
    N = len(data_array)
    fs = fs_mhz * 1e6
    dt = 1 / fs
    t = np.arange(N) * dt
    
    # Expected frequencies for annotation
    f_diff_expected = delta_nu_khz
    if sideband_name.lower() == "upper":
        f_sum_expected = (nu_lo_khz * 2) + delta_nu_khz
    else:
        f_sum_expected = (nu_lo_khz * 2) - delta_nu_khz

    # --- 2. Spectral Analysis ---
    freqs = np.fft.rfftfreq(N, d=dt)
    if_fft = np.fft.rfft(data_array)
    # Power spectrum (Normalized by N to represent Volts^2)
    power_spectrum = (np.abs(if_fft) / N)**2 

    # --- 3. Fourier Filtering (Low-Pass Filter) ---
    # We want to keep the Difference frequency and kill the Sum frequency.
    # We set the cutoff halfway between the difference and the sum.
    cutoff_hz = (f_diff_expected + 50) * 1e3 # 50kHz buffer, or adjust as needed
    
    if_fft_filtered = if_fft.copy()
    if_fft_filtered[freqs > cutoff_hz] = 0
    
    # Inverse transform to get the "Clean" Intermediate Frequency (IF)
    if_filtered_time = np.fft.irfft(if_fft_filtered)

    # --- 4. Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    
    # [Plot 1] Power Spectrum (Frequency Domain)
    axes[0].semilogy(freqs/1e3, power_spectrum, color='tab:red', lw=1.5, label="Power Spectrum")
    axes[0].set_title(f"Power Spectrum: {sideband_name} Sideband Data")
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Power (Arb. Units)")
    axes[0].grid(True, which='both', alpha=0.3)
    
    # Annotate Sum and Difference
    axes[0].axvline(f_diff_expected, color='blue', linestyle='--', label = f"Difference: {f_diff_expected} kHz")
    axes[0].axvline(f_sum_expected, color='black', linestyle='--', label = f"Sum: {f_sum_expected} kHz")
    
    harm = 0
    if harmonics == True:
        for i in range(nu_lo_khz):
            if i % f_diff_expected == 0:
                if harm == 0:
                    axes[0].axvline(i, color='green', linestyle='--', alpha=0.5, label = f"Harmonics")
                    harm+=1
                else:
                    axes[0].axvline(i, color='green', linestyle='--', alpha=0.5)
                    harm+=1
    axes[0].legend()

    # [Plot 2] Original Time Domain (Raw Data)
    axes[1].plot(t[:400]*1e6, data_array[:400], label="Raw IF Data (Mixed Signal)", color='tab:gray')
    axes[1].set_title("Time Domain: Raw Output (Sum + Difference Combined)")
    axes[1].set_xlabel("Time (μs)")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].legend(loc='upper right')

    # [Plot 3] Filtered Time Domain (The Difference Frequency)
    # This shows the "beat" result of the mixer
    axes[2].plot(t[:400]*1e6, if_filtered_time[:400], color='tab:green', lw=2, label="Filtered (Difference Only)")
    axes[2].set_title(f"Time Domain: Fourier Filtered {f_diff_expected} kHz Signal")
    axes[2].set_xlabel("Time (μs)")
    axes[2].set_ylabel("Voltage (V)")
    axes[2].legend(loc='upper right')

    fig.suptitle(f"DSB Mixer Analysis: {sideband_name} Sideband\nLO={nu_lo_khz} kHz, Δν={delta_nu_khz} kHz", fontsize=16)
    plt.show()

def analyze_ssb_mixer(I_data, Q_data, fs_mhz, target_df_khz):
    """
    Performs SSB analysis, phase correction, and complex spectral density.
    """
    # --- 1. Setup ---
    N = len(I_data)
    fs = fs_mhz * 1e6
    dt = 1 / fs
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    
    # --- 2. Phase Analysis (7.3.1) ---
    # We find the phase offset by comparing the two signals
    # We use the analytic signal via Hilbert transform or cross-correlation
    # Simple method: use the phase of the peak in the cross-power spectrum
    fft_I = np.fft.fft(I_data)
    fft_Q = np.fft.fft(Q_data)
    cross_power = fft_I * np.conj(fft_Q)
    peak_idx = np.argmax(np.abs(cross_power))
    measured_phase_diff = np.angle(cross_power[peak_idx]) # in radians
    
    # Reverting to DSB: Remove the offset so they are "identical"
    # Rotate Q to align with I
    Q_aligned = np.real(np.fft.ifft(fft_Q * np.exp(1j * measured_phase_diff)))
    
    # --- 3. The SSB Mixer (7.3.2) ---
    # Distinguish positive/negative delta_nu using original I and Q
    # Construct complex signal: Z = I + jQ
    z_ssb = I_data + 1j * Q_data
    fft_ssb = np.fft.fftshift(np.fft.fft(z_ssb))
    psd_ssb = (np.abs(fft_ssb) / N)**2

    # --- 4. The Complex IQ Result (7.3.3) ---
    # This shows the ability to distinguish sidebands
    # If delta_nu is +10kHz, we expect a peak at +10kHz and suppression at -10kHz
    
    # --- 5. Visualization ---
    fig, axes = plt.subplots(nrows=3, figsize=(15, 10))
    
    # [Plot A] Time Domain Comparison
    t_us = np.arange(N) * dt * 1e6
    axes[0].plot(t_us[:200], I_data[:200], label='I (Real)', color='blue')
    axes[0].plot(t_us[:200], Q_data[:200], label='Q (Imag)', color='red', alpha=0.7)
    axes[0].set_title(f"Time Domain: Raw IQ\nMeasured Phase: {np.degrees(measured_phase_diff):.2f}°")
    axes[0].set_xlabel("Time (μs)")
    axes[0].legend()

    # [Plot B] Reverted DSB (7.3.1)
    axes[1].plot(t_us[:200], I_data[:200], label='I', color='blue')
    axes[1].plot(t_us[:200], Q_aligned[:200], label='Q (Aligned)', color='green', linestyle='--')
    axes[1].set_title("Reverted DSB: Phase Offset Removed")
    axes[1].set_xlabel("Time (μs)")
    axes[1].legend()

    # [Plot C] Complex Spectrum (7.3.2 / 7.3.3)
    axes[2].semilogy(freqs/1e3, psd_ssb, color='purple')
    axes[2].axvline(target_df_khz, color='green', linestyle=':', label='Target +Δν')
    axes[2].axvline(-target_df_khz, color='red', linestyle=':', label='Target -Δν')
    axes[2].set_title("Complex Power Spectrum (I + jQ)")
    axes[2].set_xlabel("Frequency (kHz)")
    axes[2].set_ylabel("Power")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    print(f"Detected Phase Difference: {np.degrees(measured_phase_diff):.4f} degrees")
    return measured_phase_diff

def generate_simulated_ssb_data(N=2048, fs_mhz=2.0, delta_nu_khz=10.0, phase_offset_deg=90.0):
    """
    Generates simulated I and Q data for an SSB mixer.
    
    Args:
        N: Number of samples
        fs_mhz: Sampling rate in MHz
        delta_nu_khz: The IF frequency (beat frequency)
        phase_offset_deg: The phase shift between I and Q (Ideally 90)
        
    Returns:
        I_sim, Q_sim: Two numpy arrays of shape (2048,)
    """
    fs = fs_mhz * 1e6
    t = np.arange(N) / fs
    f_if = delta_nu_khz * 1e3
    
    # Generate I (Reference Cosine)
    I_sim = np.cos(2 * np.pi * f_if * t)
    
    # Generate Q (Phase shifted version)
    # 90 degrees makes it a Sine, but we add a small error to mimic real cables
    phase_rad = np.radians(phase_offset_deg)
    Q_sim = np.cos(2 * np.pi * f_if * t - phase_rad)
    
    # Add a tiny bit of white noise to make it realistic
    I_sim += np.random.normal(0, 0.05, N)
    Q_sim += np.random.normal(0, 0.05, N)
    
    return I_sim, Q_sim