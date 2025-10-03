# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:43:06 2023

@author: SanjayanA
"""

import re
import numpy as np
import glob
import os.path as op
import os
from scipy.io import loadmat
from configparser import RawConfigParser
import pickle


default_roi = ['LF', 'LF', 'LF', 'LF', 'LF', 'LF', 'LF', 'LF', 'Cross-L', 'Cross-L', 'LT', 'LT', 'LT', 'LT', 'LT', 'LT', 'LT', 'LT', 'RF', 'RF', 'RF', 'RF', 'RF', 'RF', 'RF', 'RF', 'Cross-R', 'Cross-R', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT']


def _open(fname):
    return open(fname, 'r', encoding='latin-1')





def read_nirx(fname, protocol = {}):
    """
    Read data from NIRX data folder.

    Parameters
    ----------
    fname : str
        Path to the folder containing NIRX data.
    protocol : dict, optional
        Dictionary mapping event IDs to event labels. Default is an empty dictionary.

    Returns
    -------
    raw : dict
        Dictionary containing data, sampling frequency, triggers, labels, wavelengths,
        channel labels, and distances.
    """
    
    
    if (fname.endswith('.pkl')):
        # Passed a pickle file so just read it into a dict and return
        with open(fname, "rb") as pickle_file:
            raw = pickle.load(pickle_file)
            return raw

    
    
    # NIRScout devices and NIRSport1 devices
    keys = ('hdr', 'inf', 'set', 'tpl', 'wl1', 'wl2', 'config.txt', 'probeInfo.mat')
    
    # Check if required files exist and store names for later use
    files = dict()
    for key in keys:
        files[key] = glob.glob('%s/*%s' % (fname, key))
        fidx = 0
        files[key] = files[key][fidx]
    
    # Read number of rows/samples of wavelength data
    with _open(files['wl1']) as fid:
        last_sample = fid.read().count('\n') - 1
    
    # Read header file
    # The header file isn't compliant with the configparser. So all the
    # text between comments must be removed before passing to parser
    with _open(files['hdr']) as f:
        hdr_str_all = f.read()
        
    hdr_str = re.sub('#.*?#', '', hdr_str_all, flags=re.DOTALL)
    hdr = RawConfigParser()
    hdr.read_string(hdr_str)
    
    fnirs_wavelengths = [int(s) for s in
                         re.findall(r'(\d+)',
                                    hdr['ImagingParameters'][
                                        'Wavelengths'])]
    
    sources = np.asarray([int(s) for s in
                          re.findall(r'(\d+)-\d+:\d+',
                                     hdr['DataStructure']
                                     ['S-D-Key'])], int)
    detectors = np.asarray([int(s) for s in
                            re.findall(r'\d+-(\d+):\d+',
                                       hdr['DataStructure']
                                       ['S-D-Key'])], int)
    
    samplingrate = float(hdr['ImagingParameters']['SamplingRate'])
    
    
    # Read information about probe/montage/optodes
    # A word on terminology used here:
    #   Sources produce light
    #   Detectors measure light
    #   Sources and detectors are both called optodes
    #   Each source - detector pair produces a channel
    #   Channels are defined as the midpoint between source and detector
    mat_data = loadmat(files['probeInfo.mat'])
    probes = mat_data['probeInfo']['probes'][0, 0]
    requested_channels = probes['index_c'][0, 0]
    src_locs = probes['coords_s3'][0, 0] / 100.
    det_locs = probes['coords_d3'][0, 0] / 100.
    ch_locs = probes['coords_c3'][0, 0] / 100.
    
    
    
    # Determine requested channel indices
    # The wl1 and wl2 files include all possible source - detector pairs.
    # But most of these are not relevant. We want to extract only the
    # subset requested in the probe file
    req_ind = np.array([], int)
    for req_idx in range(requested_channels.shape[0]):
        sd_idx = np.where((sources == requested_channels[req_idx][0]) &
                          (detectors == requested_channels[req_idx][1]))
        req_ind = np.concatenate((req_ind, sd_idx[0]))
    req_ind = req_ind.astype(int)
    
    snames = [f"S{sources[idx]}" for idx in req_ind]
    dnames = [f"_D{detectors[idx]}" for idx in req_ind]
    sdnames = [m + str(n) for m, n in zip(snames, dnames)]
    chnames = [val for pair in zip(sdnames, sdnames) for val in pair];
    wavelengths = fnirs_wavelengths*int(len(chnames)/2);
    
    # Calculate distances from chnames
    distances = [];
    for ch_n in chnames:
        sd_strs = ch_n.split('_');
        s,d = [int(a[1:])-1 for a in sd_strs]  # Due to indexing of s/d being +1
        d = np.linalg.norm(src_locs[s, :] - det_locs[d, :])
        distances.append(d)
    
    onset = [];
    description = [];
    
    # Read triggers from event file
    #files['tri'] = files['hdr'][:-3] + 'evt';
    files['tri'] = files['hdr'][:-4] + '_interspersed_disc' '.evt';

    
    if op.isfile(files['tri']):
        with _open(files['tri']) as fid:
            t = [re.findall(r'(\d+)', line) for line in fid]
            
        for t_ in t:
            binary_value = ''.join(t_[1:])[::-1]
            desc = float(int(binary_value, 2))
            trigger_frame = float(t_[0])
            onset.append(trigger_frame)
            description.append(desc)
            
    onset = np.array(onset);
    onset = onset.astype(int);
    
    # Rename description
    description = [protocol.get(str(id_), str(id_)) for id_ in description];
            
    # Read data from file
    wl1_data = np.loadtxt(files['wl1'])
    wl2_data = np.loadtxt(files['wl2'])
    
    # Filter data for channels we want
    wl1_data = wl1_data[:, req_ind].T;
    wl2_data = wl2_data[:, req_ind].T;
    
    # Combine wavelength data
    data = np.vstack([row for pair in zip(wl1_data, wl2_data) for row in pair]);
    
    # Convert to raw dict
    raw = {
          'data': data, 
          'sfreq': samplingrate, 
          'trigger_samples': onset, 
          'trigger_labels': np.array(description), 
          'wavelengths': np.array(wavelengths), 
          'ch_labels': np.array(chnames), 
          'distances': np.array(distances),
          'roi': np.array(default_roi)
    }
    
    return raw


def read_nirx_using_mne(file_name, protocol={}):
    import mne
    
    raw = mne.io.read_raw_nirx(file_name, preload=True)

    # data
    data = raw.get_data()

    # sFreq
    sfreq = raw.info['sfreq']

    # Extract triggers
    events = mne.events_from_annotations(raw)
    trigger_samples = events[0][:, 0];
    reversed_dict_labels = {value: key for key, value in events[1].items()}
    trigger_labels = [reversed_dict_labels.get(x) for x in events[0][:,2]];
    trigger_labels = [protocol.get(str(id_), str(id_)) for id_ in trigger_labels];

    # Get wavelengths used
    wavelengths = [ch['loc'][9] for ch in raw.info['chs']]

    # Get channel distances
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks='all')

    # Get channel labels (as S1_D1 etc)
    # Define a function to remove the last 4 characters from a string
    def remove_last_four(s):
        return s[:-4]

    # Vectorize the function and apply it to the numpy array
    vectorized_remove_last_four = np.vectorize(remove_last_four)
    ch_labels = vectorized_remove_last_four(raw.info.ch_names)

     # Convert to raw dict
    raw = {
           'data': data, 
           'sfreq': sfreq, 
           'trigger_samples': trigger_samples, 
           'trigger_labels': np.array(trigger_labels), 
           'wavelengths': np.array(wavelengths), 
           'ch_labels': ch_labels, 
           'distances': distances,
           'roi': np.array(default_roi)
     }
     
    return raw
 

def concat_raw_outputs(raw_list):
    # Assume the following are same, and if not, these will not be included
    # sfreq
    # wavelengths
    # ch_labels
    # distance
    
    combined_raw = raw_list[0].copy();
    time_shift_samples = combined_raw['data'].shape[1]
    
    for idx in np.arange(1, len(raw_list)):
        raw_to_add = raw_list[idx];
        
        # Check if constants are same
        if (combined_raw['sfreq'] == raw_to_add['sfreq'] and
            (combined_raw['wavelengths'] == raw_to_add['wavelengths']).all() and
            (combined_raw['ch_labels'] == raw_to_add['ch_labels']).all() and
            combined_raw['data'].shape[0] == combined_raw['data'].shape[0]):
            
            # Valid so add data, trigger samples and trigger labels 
            shifted_trig_samples = raw_to_add['trigger_samples'] + time_shift_samples
            combined_raw['trigger_samples'] = np.concatenate([combined_raw['trigger_samples'], shifted_trig_samples])
            combined_raw['trigger_labels'] = np.concatenate([combined_raw['trigger_labels'], raw_to_add['trigger_labels']])
            combined_raw['data'] = np.concatenate([combined_raw['data'], raw_to_add['data']], axis=1)
            
            time_shift_samples = raw_to_add['data'].shape[1]
            
        else:
            print("The device/settings are not the same as the others, omitting...\n")
            continue
        
    return combined_raw;


def remove_trigger_idx(raw_dict, idx_list):
    return_dict = raw_dict.copy();
    
    return_dict['trigger_samples'] = np.delete(return_dict['trigger_samples'], idx_list);
    return_dict['trigger_labels'] = np.delete(return_dict['trigger_labels'], idx_list);
    
    return return_dict;

# Export destination needs to be 
def merge_and_export(file_loc_arr, export_dest, protocols):
    rawArr = []
    for dir_path in file_loc_arr:
        # All requirements were met
        inner_folder = os.path.basename(dir_path);
        subID = os.path.basename(os.path.dirname(dir_path));
        src = f"{subID}.{inner_folder}"
        
        split_inner = inner_folder.split('.')
        if (len(split_inner) > 1):
            # IF contains more than 1 segment, then assume protocol embedded in recording label
            prot = inner_folder.split('.')[0]
        else:
            # Take from outer folder name (LEGACY => should not be used)
            print(f"======== Used legacy protocol extraction for recording {inner_folder} =========")
            prot = src.split('.')[1]
        
        rawArr.append(read_nirx(dir_path, protocol=protocols[prot]))
        
    combined_raw = concat_raw_outputs(rawArr)
    
    # Export as pickle
    with open(export_dest, "wb") as pickle_file:
        pickle.dump(combined_raw, pickle_file)


if (__name__ == '__main__'):
    # For testing
    
    fname = r'C:\Users\SanjayanA\Documents\Data\C100.S012.S013.F\2022-11-28_003';
    
    protocol = {
                '1.0':'35dB_Det',
                '2.0':'50dB_Det',
                '3.0':'65dB_Det',
                '4.0':'80dB_Det',
                '5.0':'UNUSED',
                '6.0':'UNUSED',
                '7.0':'UNUSED',
                '8.0':'UNUSED',
                '9.0':'start/stop',
                '10.0':'Nov_OFFSET',
                '11.0':'35dB_Nov',
                '12.0':'50dB_Nov',
                '13.0':'65dB_Nov'
                }
    
    output = read_nirx(fname, protocol)