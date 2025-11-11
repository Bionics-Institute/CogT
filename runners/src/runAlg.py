from fnirs_preproc.src.fnirs_preprocessing_functions import fnirs_preprocessing, spaced_random_samples
from connectivity_calcs.fnirs_graph_connectivity import  compute_fnirs_graph_metrics

import os
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'

#Merging Alg V2 into MVP dev

# For debugging with MNE (use mne for preproc instead)
#import importlib
#old = importlib.import_module("preprocessing_[legacy1]")
#fnirs_preprocessing = old.run_preproc_mne

import numpy as np
import pandas as pd
import importlib
import re
import random
from collections import defaultdict
import numpy as np

def _to_clean_str(x):
    # Convert to string safely; drop empties/None/NaN later
    s = "" if x is None else str(x)
    return s.strip()


subject_list = []


def _calculate_chromophore_means(df):
    """
    Calculate the average of epoch columns in the DataFrame for 'Hbo' and 'Hbr' chromophores.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame. Must contain columns starting with 'Epoch_' and a 'Chromophore' column. 
                           'Epoch_' columns should contain numeric data. 'Chromophore' column should contain 
                           the strings 'Hbo' and 'Hbr'.
    
    Returns:
    tuple: A tuple containing two numpy arrays. The first array contains the average of epoch columns for 
           'Hbo' chromophore. The second array contains the average of epoch columns for 'Hbr' chromophore.
    """
    
    # First, we need to identify which columns are the epoch columns.
    # These are the ones that start with 'Epoch_'.
    epoch_columns = [col for col in df.columns if col.startswith('Epoch_')]
    
    # Next, filter the DataFrame based on the 'Chromophore' column.
    hbo_df = df[df['Chromophore'] == 'HbO'][epoch_columns]
    hbr_df = df[df['Chromophore'] == 'Hbr'][epoch_columns]
    
    # Calculate the mean of the epochs for each filtered DataFrame.
    # The axis=1 parameter means that we're calculating the mean across columns (i.e., for each row).
    hbo_means = hbo_df.mean(axis=0).values
    hbr_means = hbr_df.mean(axis=0).values
    
    hbo_means = hbo_means.round(3);
    hbr_means = hbr_means.round(3);
    
    
    # Return the means as numpy arrays.
    return hbo_means, hbr_means

class InputArrayError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class SamplingFrequencyError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class triggerLabelError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class newTriggerLabelError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class triggerSampleError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class srcDetLengthError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class wavelengthError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)


class channelLabelError(ValueError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class InvalidStopObj(TypeError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class PreprocessingError(TypeError):
    """Custom exception for specific value errors.

    Attributes:
        value -- input value which caused the error
        message -- explanation of the error
    """
    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

class invalidDataError(ValueError):

    def __init__(self, value, message="Invalid value provided"):
        self.value = value
        self.message = message
        super().__init__(self.message)

def simplify_token(token):
    parts = token.split('_')
    if len(parts) == 4:
        return f"{parts[0]}_{parts[2]}_{parts[3]}"
    return token  # Return unchanged if not 4 parts


def main(inputArray,             # 2D Double Array (nChannels x nSamples)
                samplingFrequency,      # Float
                triggerLabels,          # 1D String Array (nStim x 1)
                triggerSamples,         # 1D Int Array (nStim x 1)
                srcDetDistArray,        # 1D Float Array (nChannels x 1)
                wavelengths,            # 1D Float Array (nChannels x 1)
                channelLabels):        # Object of { token1: bool, token2: bool, ... }
    
    #os.nice(20)

    triggerLabels_simple = np.array([simplify_token(tok) for tok in triggerLabels])
    triggerLabels = triggerLabels_simple
   
   
    #print(f"Source Detector Distance: {srcDetDistArray}")
    # Preprocessing & Epoch the data
    min_expect_np_channels = 4
    max_expect_np_channels = 92
    samplingfreq_lower_bound = 7
    samplingfreq_upper_bound = 18

    min_test_length = int(samplingFrequency * 18) #Some silence baseline and nonsilennce baseline triggers overlap
    trigger_Status = False
    srcDetDistArray_mm = srcDetDistArray *1000
    srcDetDist_Spec_start = 21
    srcDetDist_Spec_end = 41
    wavelength_start = 700
    wavelength_end = 900
    mean_sd_distance =np.round(np.mean(srcDetDistArray_mm),0)

    output_df = None

    ### Test Case: 1 
    if len(inputArray) < min_expect_np_channels:
        print(f"Parameter: inputArray has only  {len(inputArray)} channels; fewer than {min_expect_np_channels} no of channels.")
        raise InputArrayError(len(inputArray), f"Parameter: inputArray has only  {len(inputArray)} channels; fewer than {min_expect_np_channels} no of channels.")

    if len(inputArray) > max_expect_np_channels:
        print(f"Parameter: inputArray has only  {len(inputArray)} channels; more than     {max_expect_np_channels} no of channels.")
        raise InputArrayError(len(inputArray), f"Parameter: inputArray has only  {len(inputArray)} channels; more than     {max_expect_np_channels} no of channels.")


    ### Cant test for correct samples measured since test start; SO using this;
    print(f"Size of Input Array (mins): {inputArray.shape[1] /samplingFrequency / 60}")

    
    ### Test Case: 2
    if int(samplingFrequency) not in np.arange(samplingfreq_lower_bound, samplingfreq_upper_bound+1):
        print(f"Parameter: samplingFrequency {int(samplingFrequency)}  is outside the defined range {samplingfreq_lower_bound} and {samplingfreq_upper_bound}")
        raise SamplingFrequencyError(samplingFrequency, f"Parameter: samplingFrequency {int(samplingFrequency)}  is outside the defined range {samplingfreq_lower_bound} and {samplingfreq_upper_bound}")

            
    ### Test case 6

    if len(srcDetDistArray) > max_expect_np_channels:
        print(f"Parameter srcDetDistArray of length {len(srcDetDistArray)} is greater than max no of channels {max_expect_np_channels}")
        raise srcDetLengthError(len(srcDetDistArray), f"Parameter srcDetDistArray of length {len(srcDetDistArray)} is greater than max no of channels {max_expect_np_channels}")
       
    #print(f"Test Case #6 Source Detector Distance: {srcDetDistArray_mm}")
    srcDetDistArray_rounded = [round(num,0) for num in srcDetDistArray_mm]  
    srcdet_dist_inrange = np.intersect1d(srcDetDistArray_rounded, np.arange(srcDetDist_Spec_start, srcDetDist_Spec_end +1))

    if len(srcdet_dist_inrange) == 0:
        #print(f"Parameter srcDetDistArray distances {srcDetDistArray_mm} are outside the range {srcDetDist_Spec_start} and {srcDetDist_Spec_end}.")
        raise srcDetLengthError(srcDetDistArray_mm, f"Parameter srcDetDistArray distances {srcDetDistArray_mm} are outside the range {srcDetDist_Spec_start} and {srcDetDist_Spec_end}.")
    
    ### Test Case 7

    #print(f"Test Case #7 Wavelengths: {wavelengths}")

    wavelengths_rounded = [round(num,0) for num in wavelengths]  
    wavelengths_inrange = np.intersect1d(wavelengths_rounded, np.arange(wavelength_start, wavelength_end +1))

    #print(f"Test Case #8 Wavelengths: {wavelengths_inrange}")
   
    if len(wavelengths_inrange) == 0:
        print(f"Parameter Wavelength distances {wavelengths_rounded} are outside the range {wavelength_start} and {wavelength_end}.")
        raise wavelengthError(wavelengths_rounded, f"Parameter Wavelength distances {wavelengths_rounded} are outside the range {wavelength_start} and {wavelength_end}.")
    
   
    #print(f"16. Trigger labels before preproc: {triggerLabelsModified}")
    try:
        stim_df, data_hb, chromophore_labels = fnirs_preprocessing(inputArray, 
                                    samplingFrequency, 
                                    triggerSamples, 
                                    triggerLabels, 
                                    wavelengths, 
                                    channelLabels, 
                                    srcDetDistArray)
        

    except(ValueError, ArithmeticError, TypeError, ArithmeticError) as e:
        raise PreprocessingError(f"Pre-processing Exception: Invalid data or data type, {e}")
    
    # Just get HbO values
    unique_values, first_idx = np.unique(channelLabels, return_index=True)
    order = np.argsort(first_idx)
    unique_arr = unique_values[order]
    first_idx = first_idx[order]
    data_hbo = data_hb[first_idx,:]

    #Get dlPFC Subset

    endings = ("_D1", "_D3", "_D4")
    unique_arr = np.array(list(dict.fromkeys(channelLabels)))

    mask = np.array([lbl.endswith(endings) for lbl in unique_arr])
    subset = unique_arr[mask]

    print("Matching entries:", subset)

    indices = np.where(mask)[0]
    print("Indices:", indices)
    print("Subset:", unique_arr[indices])
    data_hbo_dlpfc = data_hbo[indices,:]
    vec_names = unique_arr[indices]


    metrics, corr, vec, (i, j) = compute_fnirs_graph_metrics(data_hbo_dlpfc, fs=None, band=(0.01, 0.08), detrend=False, use_abs=True)
# corr is NxN, vec_names is length N (e.g. ['S4_D1', 'S4_D3', ...])

    n = corr.shape[0]

    # get upper-triangular indices excluding diagonal
    r, c = np.triu_indices(n, k=1)

    # corresponding source-detector name pairs
    pair_names = [(vec_names[i], vec_names[j]) for i, j in zip(r, c)]

    # corresponding correlation values
    corr_vals = corr[r, c]

    # if you just want to print them:
    for (ch1, ch2), val in zip(pair_names, corr_vals):
        print(f"{ch1}  --  {ch2}: {val:.3f}")

    df_corr = pd.DataFrame({
    "channel_pair1": [vec_names[i] for i in r],
    "channel_pair2": [vec_names[j] for j in c],
    "corr": corr_vals})



        # write outdf to csv
        # Outputs
    return df_corr