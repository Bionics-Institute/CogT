# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:16:30 2023

Reimplementation of https://github.com/mne-tools/mne-python/tree/maint/0.21/mne/preprocessing/nirs

@author: SanjayanA
"""

import numpy as np    
from scipy.signal import butter, filtfilt, iirfilter, sosfilt, sosfiltfilt, sosfilt_zi
from scipy import linalg
from scipy.io import loadmat
from scipy.interpolate import interp1d
import pandas as pd
import re

# If called from same script or a script in the same directory
if __name__=='__main__' or __name__=='fnirs_preprocessing_functions':
    from haem_conc import molar_extinct_coeff
else:
    # Remove dot if calling file from another folder
    from .haem_conc import molar_extinct_coeff


def optical_density(wavelength_data):
    """
    Convert wavelength data to optical density.

    """
    # Copy
    optic_density_data = np.copy(wavelength_data)

    #%% Given light intensities should be >= 0, we will correct for any irregularities
    # Get rid of negative values
    np.abs(optic_density_data, out=optic_density_data)

    # Find the minimum positive value in the array
    min_positive = np.min(optic_density_data[optic_density_data > 0])

    # Replace zero values with the minimum positive value
    optic_density_data[optic_density_data == 0] = min_positive
    
    
    #%% Calculate optical density using the negative logarithm (base 10) of the intensity ratio
    # Divide each channel by its mean
    channel_means = np.mean(optic_density_data, axis=1, keepdims=True)
    optic_density_data /= channel_means

    # Apply logarithm and multiply by -1
    optic_density_data = -np.log(optic_density_data)
    
    return optic_density_data


def tddr(data, sample_rate):
    """
    Apply the TDDR algorithm on a 2D array of dimensions nChannels x nSamples.
    Adapted using ChatGPT-4 using https://github.com/frankfishburn/TDDR/ as src

    Parameters:
    data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples.
    sample_rate (float): The sample rate of the data in Hz.

    Returns:
    corrected_data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples with TDDR applied.
    """
    # Preprocess: Separate high and low frequencies
    filter_cutoff = 0.5
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - data_mean

    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        data_low = filtfilt(fb, fa, data_centered, axis=1, padlen=0)
    else:
        data_low = data_centered

    data_high = data_centered - data_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(data.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(data_low, axis=1)

    # Step 2. Initialize observation weights
    w = np.ones(deriv.shape)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:

        iter = iter + 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv, axis=1, keepdims=True) / np.sum(w, axis=1, keepdims=True)

        # Step 3b. Calculate absolute residuals of estimate
        dev = np.abs(deriv - mu)

        # Step 3c. Robust estimate of standard deviation of the residuals
        sigma = 1.4826 * np.median(dev, axis=1, keepdims=True)

        # Step 3d. Scale deviations by standard deviation and tuning parameter
        r = dev / (sigma * tune)

        # Step 3e. Calculate new weights according to Tukey's biweight function
        w = ((1 - r**2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within machine-precision of old estimate
        if np.all(np.abs(mu - mu0) < D * np.maximum(np.abs(mu), np.abs(mu0))):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    data_low_corrected = np.cumsum(np.concatenate((np.zeros((data.shape[0], 1)), new_deriv), axis=1), axis=1)

    # Postprocess: Center the corrected signal
    data_low_corrected = data_low_corrected - np.mean(data_low_corrected, axis=1, keepdims=True)

    # Postprocess: Merge back with uncorrected high frequency component
    corrected_data = data_low_corrected + data_high + data_mean

    return corrected_data


def filter_data(data, sfreq, l_freq=None, h_freq=None, order=3, filter_type='butter',
                l_trans_bandwidth=None, h_trans_bandwidth=None):
    """
    Filter a 2D numpy array of dimensions nChannels x nSamples using an IIR filter.

    Parameters:
    data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples.
    sfreq (float): The sampling frequency of the data.
    l_freq (float or None): The low cutoff frequency. If None, no lowpass filtering is applied.
    h_freq (float or None): The high cutoff frequency. If None, no highpass filtering is applied.
    order (int): The order of the filter.
    filter_type (str): The type of the filter. Default is 'butter' for Butterworth filter.
    l_trans_bandwidth (float or None): The low transition bandwidth. If None, the filter design method will use its default value.
    h_trans_bandwidth (float or None): The high transition bandwidth. If None, the filter design method will use its default value.

    Returns:
    filtered_data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples with the filter applied.
    """

    if l_freq is None and h_freq is None:
        raise ValueError("Both l_freq and h_freq cannot be None. Specify at least one cutoff frequency.")

    if l_freq is not None and h_freq is not None:
        if l_freq > h_freq:
            raise ValueError("l_freq must be less than h_freq.")

    filtered_data = np.copy(data)

    if l_freq is not None and h_freq is not None:
        btype = 'band'
        Wn = (l_freq / (sfreq / 2), h_freq / (sfreq / 2))
    elif l_freq is not None:
        btype = 'high'
        Wn = l_freq / (sfreq / 2)
    else:
        btype = 'low'
        Wn = h_freq / (sfreq / 2)

    if l_trans_bandwidth is not None or h_trans_bandwidth is not None:
        if btype == 'band':
            # Compute the stopband edges
            f_stop_low = l_freq - l_trans_bandwidth / 2
            f_stop_high = h_freq + h_trans_bandwidth / 2
            Wn = [(f_stop_low / (sfreq / 2)), (f_stop_high / (sfreq / 2))]
        else:
            # Compute the normalized cutoff frequency
            if btype == 'low':
                f_cutoff = l_freq
                f_trans_bandwidth = l_trans_bandwidth
            elif btype == 'high':
                f_cutoff = h_freq
                f_trans_bandwidth = h_trans_bandwidth
            else:
                raise ValueError("Invalid filter type")
            Wn = (f_cutoff / (sfreq / 2))
        sos = iirfilter(order, Wn=Wn, rp=1, rs=60, btype=btype, output='sos', ftype=filter_type)
    else:
        sos = iirfilter(order, Wn, btype=btype, ftype=filter_type, output='sos')

    for ch in range(filtered_data.shape[0]):
        filtered_data[ch] = sosfiltfilt(sos, filtered_data[ch], padlen=_esimate_pad_len(sos))
        
    return filtered_data


def _esimate_pad_len(sos_coeffs, max_samples=100000):
    """
        Estimate the number of ringing samples in a filter's impulse response,
        given its second-order sections (SOS) coefficients.
    
        Parameters
        ----------
        sos_coeffs : numpy.ndarray
            The array containing the SOS filter coefficients, with each row
            representing a second-order section [b0, b1, b2, a0, a1, a2].
        max_samples : int, optional
            The maximum number of samples to consider when estimating ringing
            samples (default is 100000).
    
        Returns
        -------
        ringing_samples : int
            An approximation of the number of ringing samples in the filter's
            impulse response.
    """
    z_initial = np.zeros((len(sos_coeffs), 2))
    samples_per_chunk = 1000
    num_chunks = int(np.ceil(max_samples / samples_per_chunk))
    input_signal = np.zeros(samples_per_chunk)
    input_signal[0] = 1
    last_significant = samples_per_chunk
    threshold = 0

    for chunk_idx in range(num_chunks):
        output_signal, z_initial = sosfilt(sos_coeffs, input_signal, zi=z_initial)
        input_signal[0] = 0  # Reset first element to zero for following iterations
        abs_output = np.abs(output_signal)
        threshold = np.maximum(0.001 * np.max(abs_output), threshold)
        significant_indices = np.where(abs_output > threshold)[0]

        if significant_indices.size > 0:
            last_significant = significant_indices[-1]
        else:
            ringing_samples = (chunk_idx - 1) * samples_per_chunk + last_significant
            break
    else:
        ringing_samples = samples_per_chunk * num_chunks

    return ringing_samples




def compute_sci(data, sfreq, l_freq=0.7, h_freq=1.5, l_trans_bandwidth=0.3, h_trans_bandwidth=0.3):
    """
    Compute the spatial correlation index (SCI) for the input 2D data array (nChannels x nSamples) after filtering it.
 
    Parameters:
    data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples.
    sfreq (float): The sampling frequency of the data.
    l_freq (float or None): The low cutoff frequency. If None, no lowpass filtering is applied.
    h_freq (float or None): The high cutoff frequency. If None, no highpass filtering is applied.
    l_trans_bandwidth (float or None): The low transition bandwidth. If None, the filter design method will use its default value.
    h_trans_bandwidth (float or None): The high transition bandwidth. If None, the filter design method will use its default value.
 
    Returns:
    sci (numpy.ndarray): A 1D numpy array of length nChannels containing the spatial correlation index for each channel pair.
    """
    n_channels = data.shape[0]
    filtered_data = filter_data(
        data,
        sfreq,
        l_freq,
        h_freq,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
    )

    zero_mask = np.std(data, axis=-1) == 0

    sci = np.zeros(n_channels)
    for ii in range(0, n_channels, 2):
        with np.errstate(invalid="ignore"):
            c = np.corrcoef(filtered_data[ii], filtered_data[ii + 1])[0][1]
        if not np.isfinite(c):  # someone had std=0
            c = 0
        sci[ii] = c
        sci[ii + 1] = c
    sci[zero_mask] = 0
    return sci



def beer_lambert_law(data, sfreq, ppf, wavelengths, distances, ch_labels):
    """
    Convert NIRS optical density data to haemoglobin concentration.

    Parameters:
    data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples containing the optical density data.
    sfreq (float): The sampling frequency of the data.
    ppf (float): The partial pathlength factor.
    wavelengths (numpy.ndarray): A 1D numpy array containing the wavelengths used for the NIRS measurements for each channel.
    distances (numpy.ndarray): A 1D numpy array containing the source-detector distances for each channel.
    ch_labels (numpy.ndarray): A 1D numpy array containing the channel location labels (i.e. S1_D1, S1_D2 etc.)
    
    Returns:
    hb_data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples containing the haemoglobin concentration data.
    chromophore_labels (numpy.ndarray): A 1D numpy array of dimensions nChannels x 1 containing the labels for each channel (hbo or hbr).
    """

    bad = ~np.isfinite(distances)
    bad |= distances <= 0
    if bad.any():
        print('Source-detector distances are zero on NaN, some resulting concentrations will be zero. Consider setting a montage.')
    distances[bad] = 0.

    if (distances > 0.1).any():
        print('Source-detector distances are greater than 10 cm. Large distances will result in invalid data, and are likely due to optode locations being stored in a unit other than meters.')

    hb_data = np.copy(data)
    chromophore_labels = np.empty(data.shape[0], dtype=object)
    
    for ch_name in np.unique(ch_labels):
        # Find location of current src/det pair
        ch_idx = np.where(ch_labels == ch_name)[0]; 
        
        # Extract corresponding wavelengths and distances
        pair_wavelengths = wavelengths[ch_idx];
        pair_distances = distances[ch_idx];
        
        # Get coefficient
        abs_coef = _load_absorption(pair_wavelengths);
        
        # Calculate the extinction length matrix
        EL = abs_coef * pair_distances * ppf

        # Invert the extinction length matrix
        iEL = linalg.pinv(EL)

        # Apply Beer-Lambert law and scale the data
        hb_data[ch_idx] = iEL @ hb_data[ch_idx] * 1e-3

        # Assign chromophore labels
        chromophore_labels[ch_idx] = ['HbO', 'Hbr']
        

    return hb_data, chromophore_labels



def _load_absorption(freqs):
    """Load molar extinction coefficients."""
    interp_hbo = interp1d(molar_extinct_coeff[:, 0], molar_extinct_coeff[:, 1], kind='linear')
    interp_hb = interp1d(molar_extinct_coeff[:, 0], molar_extinct_coeff[:, 2], kind='linear')

    ext_coef = np.array([[interp_hbo(freqs[0]), interp_hb(freqs[0])],
                         [interp_hbo(freqs[1]), interp_hb(freqs[1])]])
    abs_coef = ext_coef * 0.2303

    return abs_coef


def epoch_data(data, trigger_samples, trigger_labels, tmin, tmax, sfreq, detrend=None, baseline=(None, 0)):
    """
    Epoch the data based on trigger samples and time range.

    Parameters:
    data (numpy.ndarray): A 2D numpy array of dimensions nChannels x nSamples containing the data.
    trigger_samples (numpy.ndarray): A 1D numpy array containing the sample indices to epoch at.
    trigger_labels (numpy.ndarray): A 1D numpy array containing the trigger labels to identify the epochs.
    tmin (float): The time before the trigger to include in the epoch (negative value).
    tmax (float): The time after the trigger to include in the epoch.
    sfreq (float): The sampling frequency of the data.
    detrend (bool, optional): If True, detrend the epoch data. Default is True.
    baseline (tuple, optional): The time range for baseline correction. Default is (None, 0).

    Returns:
    epochs_df (pandas.DataFrame): A 2D pandas DataFrame containing the epoch data and metadata.
    """
    
    # Check trigger samples array to ensure all triggers can be fully epoched
    # Convert tmin and tmax to samples
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)
    # Calculate start and end samples for each trigger
    start_samples = np.array(trigger_samples) + tmin_samples
    end_samples = np.array(trigger_samples) + tmax_samples
    # Filter valid triggers that are not too early or too late in the data
    valid_indices = np.logical_and(start_samples >= 0, end_samples <= data.shape[1])
    trigger_samples = np.array(trigger_samples)[valid_indices]
    trigger_labels = np.array(trigger_labels)[valid_indices]
    
    n_channels = data.shape[0]
    epoch_length = tmax_samples - tmin_samples;
    epochs = np.empty((len(trigger_samples), n_channels, epoch_length))
    time_vector = np.linspace(tmin, tmax, epoch_length);
    time_vector = time_vector.round(decimals=3); 

    for i, trigger_sample in enumerate(trigger_samples):
        start_sample = int(trigger_sample + tmin_samples)
        end_sample = int(trigger_sample + tmax_samples)
        epoch = data[:, start_sample:end_sample]
        
        if baseline is not None:
            start, end = baseline

            if start is None:
                start = 0
            else:
                # Convert start (seconds) into start (samples)
                start = int((start - tmin)*sfreq);
            if end is None:
                end = epoch.shape[1]
            else:
                # Convert end (seconds) into end (samples)
                end = int((end - tmin)*sfreq);

            baseline_data = epoch[:, start:end]
            baseline_mean = np.mean(baseline_data, axis=1, keepdims=True)

            epoch = epoch - baseline_mean

        if detrend is not None:
            if detrend == 0:
                epoch = epoch - np.mean(epoch, axis=1, keepdims=True)
            elif detrend == 1:
                linear_trend = np.linspace(0, 1, epoch.shape[1])
                mean_trend = np.mean(epoch, axis=1, keepdims=True)
                epoch = epoch - mean_trend * linear_trend

            
        epochs[i] = epoch;
    
    
    # Broadcast epochs + metadata into pandas dataframe
    n_epochs = len(trigger_samples)
    
    # Calculate all_token_counts and this_token_counts and speech type
    all_token_counts = np.repeat(np.arange(1, n_epochs + 1), n_channels)
    
    thisTokenCountDict = {};
    thisTokenCount = [];
    speechType = [];
    for trig in trigger_labels:
        trig = re.sub(r'_[^_]*-[^_]*_', '_', trig)
        if (trig in thisTokenCountDict):
            thisTokenCountDict[trig] = thisTokenCountDict[trig] + 1; # Increment count
        else:
            thisTokenCountDict[trig] = 1;
        
        # Add token count to arr
        thisTokenCount.append(thisTokenCountDict[trig]);
        
        # Determine the speech type
        spTy = trig[-3:];
        if (spTy == 'Det'):
            speechType.append("Detection");
        elif (spTy == 'Nov'):
            speechType.append("Discrimination");
        elif (spTy == 'FSD'):
            speechType.append("Detection");
        else:
            speechType.append("Unknown");
    
    this_token_counts = np.repeat(thisTokenCount, n_channels); # Repeat for all channels
    speech_type = np.repeat(speechType, n_channels); # Repeat for all channels
    
    # Create DataFrame
    meta_df = pd.DataFrame({
        'Condition': np.repeat(trigger_labels, n_channels),
        'Channel': np.tile(np.arange(n_channels)+1, n_epochs), 
        'OnsetSample': np.repeat(trigger_samples, n_channels),
        'ThisTokenCount': this_token_counts,
        'AllTokenCount': all_token_counts,
        "SpeechType": speech_type
    })
    
    # Add Epoch and TimeVec columns to DataFrame
    numSamples = epochs.shape[2];
    epochDataMerged = np.reshape(epochs, (1,-1,numSamples))[0,:,:];
    epochDataMerged = epochDataMerged.round(decimals=3);
    numRows = epochDataMerged.shape[0];
    epochs_df = pd.DataFrame(epochDataMerged, columns=['Epoch_'+str(num) for num in np.arange(1, numSamples+1)]);
    timeNpArr = np.tile(time_vector, (numRows, 1));
    timeSampl = timeNpArr.shape[1];
    time_df = pd.DataFrame(timeNpArr, columns=['TimeVec_'+str(num) for num in np.arange(1, timeSampl+1)]);
    
    # Concat all
    epochs_df = pd.concat([epochs_df, time_df, meta_df], axis=1)
    
    
    return epochs_df



def spaced_random_samples(min_value, max_value, num_samples, min_distance):
    """
    Generate a specified number of random integers within a given range, ensuring a minimum distance 
    between each value. This function generates random integers such that each value is at least 
    `min_distance` apart from the next value in the sequence. 

    Parameters
    ----------
    min_value : int
        The minimum value (inclusive) in the range for generating random integers.
    max_value : int
        The maximum value (inclusive) in the range for generating random integers.
    num_samples : int
        The number of random integers to generate.
    min_distance : int
        The minimum distance required between the generated random integers.

    Returns
    -------
    result : numpy.ndarray
        A sorted 1D numpy array containing the generated random integers.

    Notes
    -----
    This function guarantees that each sample is at least `min_distance` apart from the next sample in the 
    sequence, but not necessarily from all other samples. If the `min_distance` requirement cannot be satisfied 
    due to the specified range or number of samples, the function will still attempt to generate the specified 
    number of samples by taking a random subset of the generated sequence, then reshuffling and repeating until 
    enough samples have been generated.
    """

    # Generate a sequence of numbers that satisfy the minimum distance condition
    sequence = np.arange(min_value, max_value + 1, min_distance)

    # Create a blank array to store the result
    result = np.zeros(num_samples, dtype=int)

    # Use a simple counter to track how many samples have been generated
    count = 0

    while count < num_samples:
        # Randomly shuffle the sequence
        np.random.shuffle(sequence)

        # Determine how many samples to take in this loop
        samples_to_take = min(len(sequence), num_samples - count)

        # Add the samples to the result array
        result[count : count + samples_to_take] = sequence[:samples_to_take]

        # Update the counter
        count += samples_to_take

    # Sort the result array
    result.sort()

    return result



# Constants
def return_early(data, name):
    """
     Used to debug fnirs_preprocessing function. Just place anywhere to save 
     data
     
     i.e.
     return_early(data_filtered, "data/mine_outputs/filter.csv")
    """
    np.savetxt(name, data, delimiter=',');
    raise Exception("Exit early")
        

def fnirs_preprocessing(data, sfreq, trigger_samples, trigger_labels, wavelengths, ch_labels, distances, roi, epoch_bounds=[-3, 27]):
    """
    Preprocess fNIRS data and return a pandas DataFrame with additional metadata columns.

    Parameters
    ----------
    data : 2D numpy array, shape (nChannels, nSamples)
        The raw fNIRS data.
    sfreq : float
        The sampling frequency of the data.
    trigger_samples : 1D numpy array, shape (M,)
        The samples at which the triggers occurred.
    trigger_labels : 1D numpy array, shape (M,)
        The labels corresponding to the trigger_samples.
    wavelengths : 1D numpy array, shape(nChannels,)
        The wavelengths used for fNIRS measurement for each channel
    ch_labels: 1D numpy array , shape (nChannels,)
        The labels corresponding to each channel location. (i.e. S1_D1, S1_D2)
    distances : 1D numpy array, shape (nChannels,)
        The source-detector distances for each channel.
    roi : 1D numpy array, shape (nChannels,)
        Region of Interest (ROI) string labels for each channel (e.g. 'rf').
        
    Returns
    -------
    epochs_df : pandas DataFrame
        The preprocessed fNIRS data in a pandas DataFrame with additional metadata columns.
        Columns include the epoch data (Epoch1, Epoch2, ...), time stamps for each epoch data
        (TimeVec1, TimeVec2, ...), trigger label (Condition), channel number (Channel),
        onset sample (OnsetSample), specific trigger count (ThisTokenCount),
        all trigger count (AllTokenCount), channel SCI (ChannelSCI), chromophore type (Chromophore),
        region of interest (ROI), and speech type (SpeechType).
    
    data_filtered : 2D numpy array, shape (nChannels, nSamples)
        The preprocess fNIRS data just before being converted with the beer-lambert law
    """

    #print(f"16a: trigger labels into preproc : {trigger_labels}")
    data[data == 0] = 1e-16
    # 1. Optical Density
    data_od = optical_density(data)
    #print(f"Preproc 1: {data_od.shape}")
    
    # 1.5. Compute SCI
    channel_sci = compute_sci(data_od, sfreq)

    #print(f"Preproc 2: {channel_sci.shape}")


    # 2. TDDR
    data_tddr = tddr(data_od, sfreq)

    #print(f"Preproc 3: {data_tddr.shape}")


    

    # 3. Filtering
    data_filtered = filter_data(data_tddr, sfreq, l_freq=None, h_freq=0.25, order=8,
                                filter_type='butter');
    
    #print(f"Preproc 4 lpass: {data_filtered.shape}")

    #print(f"Preproc 4a lpass out: {np.mean(data_filtered,axis=0)}")
    if data_filtered.shape[1] > 1000:
        data_filtered = filter_data(data_filtered, sfreq, l_freq=0.01, h_freq=None, order=8,
                                filter_type='butter')
        
    else:
        data_filtered -= data_filtered.mean(axis=1, keepdims=True)


    #print(f"Preproc 5 hpass: {data_filtered.shape}")


    # 4. Beer-Lambert Law
    data_hb, chromophore_labels = beer_lambert_law(data_filtered, sfreq, 0.1*10**-6, wavelengths, distances, ch_labels)

    #print(f"Preproc 6 hpass: {data_hb.shape}")





    
    
    # 5. Epoch Data
    tmin, tmax = epoch_bounds;
    epochs_df = epoch_data(data_hb, trigger_samples, trigger_labels, tmin, tmax, sfreq, detrend=None, baseline=(None, 0))
    
    #print(f"Preproc 6: {epochs_df.shape}")


    # Add Meta Data Columns
    num_rows = epochs_df.shape[0];    
    epochs_df["ChannelSCI"] = np.resize(channel_sci, num_rows);
    epochs_df["Chromophore"] = np.resize(chromophore_labels, num_rows)
    epochs_df["ROI"] = np.resize(roi, num_rows);
    
    
    # 6. Epoch Filtering
    # Get the epoch columns
    epoch_columns = [col for col in epochs_df.columns if col.startswith("Epoch")]
    # Create a copy of the DataFrame to not modify the original DataFrame
    df_copy = epochs_df.copy()

    #print(f"Preproc 7: {df_copy.shape}")

    # Apply the conditions only on the epoch_columns
    #df_copy[epoch_columns] = df_copy[epoch_columns].applymap(lambda x: x if -85 < x < 85 else np.nan)
    # Create a mask that only includes rows where all epoch_columns are not NaN
    mask = df_copy[epoch_columns].notna().all(axis=1)
    # Apply the mask to df_copy
    epochs_df = df_copy[mask]
    #epochs_df = epochs_df[epochs_df["ChannelSCI"] >= 0.7].dropna()

    #print(f"16 b. Epochs DF: {epochs_df}")
    

    return epochs_df, data_hb, chromophore_labels

