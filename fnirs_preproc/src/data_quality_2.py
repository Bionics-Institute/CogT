# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:09:07 2025

@author: dmao
PSP measure based on MNE-NIRS _peak_power function
"""

import numpy as np
import pandas as pd 
import json
from scipy.signal import periodogram
from scipy.stats import pearsonr

import mne
import mne_nirs

import tkinter as tk
from tkinter import filedialog

import re

import matplotlib.pyplot as plt

from simulate_recordings_3 import simulate_exData

### DATA QUALITY SPECIFIC FUNCTIONS

def def_window(raw, window_length, window_step):
    
    window_length = int(np.round(window_length * raw.info["sfreq"]))
    window_step = int(np.round(window_step * raw.info["sfreq"]))
    
    win_params = {
        'win_start_indices' : [],
        'window_length' : window_length, # in samples
        'window_step' : window_step
        }
    
    full_samples = len(raw) # how long the full dataset is
    start_sample = 0 # the start index of each window
    
    while start_sample + window_length <= full_samples:
        
        win_params['win_start_indices'].append(start_sample)
        start_sample = start_sample + window_step  # shift the start of the next window by window_step samples
        
    return win_params

def peak_power(data, win_params):
    
    orig_data = data.copy()
    filtered_data = frequency_band_selection(orig_data, freqband = 'infant')
    
    data_array = filtered_data.get_data();
    data_quality = np.zeros([len(data.ch_names),len(win_params['win_start_indices'])])
    
    for i,w in enumerate(win_params['win_start_indices']):
        for chan in range(0, len(data.ch_names), 2):
            c1 = data_array[chan, w:(w+win_params['window_length'])]
            c2 = data_array[chan + 1, w:(w+win_params['window_length'])]
    
            # protect against zero
            c1 = c1 / (np.std(c1) or 1)
            c2 = c2 / (np.std(c2) or 1)
    
            c = np.correlate(c1, c2, "full")
            c = c / (win_params['window_length'])
            [f, pxx] = periodogram(c, fs=data.info["sfreq"], window="hamming")
    
            this_data_quality = max(pxx)
            data_quality[chan, i] = this_data_quality
            data_quality[chan+1, i] = this_data_quality
                
    return data_quality

def sci(data, win_params):
    
    orig_data = data.copy()
    filtered_data = frequency_band_selection(orig_data, freqband = 'infant')
    
    data_array = filtered_data.get_data();
    data_quality = np.zeros([len(data.ch_names),len(win_params['win_start_indices'])])
    
    for i,w in enumerate(win_params['win_start_indices']):
        for chan in range(0, len(data.ch_names), 2):
            c1 = data_array[chan, w:(w+win_params['window_length'])]
            c2 = data_array[chan + 1, w:(w+win_params['window_length'])]
    
            c = np.corrcoef(c1, c2)[0,1] # Take the off-diagonal (correlation between c1 and c2)
            data_quality[chan, i] = c
            data_quality[chan+1, i] = c
                
    return data_quality

def avg_correlation(data, win_params):
    
    data_array = data.get_data();
    data_array = np.diff(data_array)
    
    data_quality = np.zeros([len(data.ch_names),len(win_params['win_start_indices'])])
    
    for i,w in enumerate(win_params['win_start_indices']):
        this_corr = np.corrcoef(data_array[range(0, len(data.ch_names),2), w:(w+win_params['window_length'])]) # Just one wavelength (?)
        this_avg_corr = np.mean(this_corr,0)
        
        for j,chan in enumerate(range(0, len(data.ch_names), 2)):
            
            data_quality[chan, i] = this_avg_corr[j]
            data_quality[chan+1, i] = this_avg_corr[j]
                
    return data_quality


def induce_bounce(data, win_params):
    data_filtered = data.filter(l_freq=0.5, h_freq=None);
    data_array_filtered = data_filtered.get_data();

    data_quality = np.zeros([len(data.ch_names),len(win_params['win_start_indices'])])
    bounce = 0.075
    half_length = int(np.floor(win_params['window_length']/2))
    
    for i, w in enumerate(win_params['win_start_indices']):        
        
        for chan in range(0, len(data.ch_names), 2):
        #Split the window in two. Add a bounce to each window.
            first_window = data_array_filtered[chan, w:(w+half_length)].copy()
            second_window = data_array_filtered[chan, (w+half_length):(w+half_length*2)].copy()
            first_window[int(half_length/2):]+=bounce
            second_window[int(half_length/2):]+=bounce
        
            data_quality[chan,i],_ = pearsonr( first_window, second_window )              
            data_quality[chan+1,i] = data_quality[chan,i]
                
    return data_quality


#JAMAL'S VERSION
# def induce_bounce(data, win_params):
#     data_filtered = data.filter(l_freq=0.5, h_freq=None);
#     data_array_filtered = data_filtered.get_data();
#     data_array = data.get_data();
#     data_array = np.diff(data_array)
#     rec_quality = np.zeros([1, len(win_params['win_start_indices'])])
#     chs_quality = np.zeros([int(len(data.ch_names)/2),len(win_params['win_start_indices'])])
#     chs_quality[:,0] = np.nan
#     bounce = 0.075
#     past_win = []
    
#     for i, w in enumerate(win_params['win_start_indices']):        
#         this_corr = np.corrcoef(data_array[range(0, len(data.ch_names),2), w:(w+win_params['window_length'])]) # Just one wavelength (?)
#         rec_quality[0,i] = np.mean( this_corr[np.triu_indices_from(this_corr, k=1)])
#         #
#         current_win = data_array_filtered[range(0, len(data.ch_names),2), w:(w+win_params['window_length'])]
#         current_win[:,int(win_params['window_length']/2):]+=bounce
#         if i>0:           
#             for j,chan in enumerate(range(0, len(data.ch_names), 2)):
#                 chs_quality[j,i],_ = pearsonr( current_win[j,:] , past_win[j,:] )              
            
#         past_win = current_win
                
#     return rec_quality, chs_quality

### AUX FUNCTIONS

def frequency_band_selection(raw, freqband = 'infant'):
    
    if freqband == 'adult':
        l_freq = 0.7; h_freq = 1.5
    elif freqband == 'infant':
        l_freq = 1.5; h_freq = 3.3
    elif freqband == 'highpass':
        l_freq = 0.5; h_freq = None;
    else:
        raise Exception("freqband selection does not match available options.")
        
    filtered = raw.filter(l_freq=l_freq,
                        h_freq=h_freq,
                        verbose=False,
                        method='iir',
                        iir_params = dict(order=8,
                                          ftype='butter',
                                          output='sos'));
    return filtered

def rms(data_in):
    if np.any(np.isnan(data_in)):
        return None
    else:
        return np.sqrt(np.mean(np.square(data_in)))


### MAIN PROCESSING FUNCTIONS

def data_to_epoch_table(data, window_parameters, data_quality_info):
    
    def getROI(channelArr, montage): # CHANNEL DEFINITIONS
        returnROI = [];
        if montage == 'EG_infant_8x8':
            for c in channelArr:
                if (c in [1,2,3,4,5,6,7,8,9,10]):
                    returnROI.append('LPF');
                elif (c in [11,12,13,14,15,16,17,18]):
                    returnROI.append('LT');
                elif (c in [19,20,21,22,23,24,25,26,27,28]):
                    returnROI.append('RPF');
                elif (c in [29,30,31,32,33,34,35,36]):
                    returnROI.append('RT');
                else:
                    returnROI.append('UNKNOWN');
        return returnROI;

    def getChromophore(channelArr, montage):
        returnChrom = [];
        if montage == 'EG_infant_8x8':
            for c in channelArr:
                if (c % 2 == 0):
                    returnChrom.append('Hbr');
                else:
                    returnChrom.append('HbO');
        return returnChrom;
    
    with open(r'R:\HUM\Projects\fNIRS\EG_Protocols\protocols.json', 'r') as f:
        all_trig_Dicts = json.load(f)
        
    trigDict = all_trig_Dicts['S008']
    # trigDict = all_trig_Dicts['S012_S013']
    
    relevantTriggers = {key: trigDict[key] for key in trigDict if (key in data.annotations.description)}; # filter triggers so that dictionary only contains triggers in data
    for old,new in relevantTriggers.items():
        data.annotations.description = [(new if old == d else d) for d in data.annotations.description];

    data.annotations.description = np.array(data.annotations.description);
        
    data.annotations.delete(np.logical_or(
        data.annotations.description == 'start/stop',
        data.annotations.description == 'Discrim_novel_offset'));
        # data.annotations.description == 'stimuli_offset'));
    
    events, eventDict = mne.events_from_annotations(data);
    
    epochData = mne.Epochs(
         data,                      # use the haemo data for epoching
         events,                         # uses the stimulus onset timings as defined in the raw_haemo file (raw.annotations.onset) 
         #reject=reject_criteria,        # not rejecting any epochs
         reject_by_annotation = False,   # annotations are not used for rejection
         proj=False,                     # projectors (which remove artifacts or improve signal quality) are not applied
         tmin = -3,                 # using the time specified in the PARAMETERS section for pretime - this is the pre-stim period to use
         tmax = 27,                # using the time specified in the PARAMETERS section for posttime - this is the post-stim period to use
         baseline=(None, 0),             # baseline correction will be applied from the start of the epoch until time zero.
         detrend=None,                   # no detrending
         preload=True,                   # all epochs are loaded into memory, which can make subsequent processing faster at the cost of higher memory usage
         verbose=False);                 # reduces the information about the processing reported in the console
    
    epochData3D = np.array(epochData.get_data());

    # Flatten 3D array to merge epoch & channel dimensions so that the new rows are:
    # E1C1, E1C2, E1C3...E2C1 etc. (epoch outerlayer, channel innerlayer)
    numSamples = epochData3D.shape[2];
    epochDataMerged = np.reshape(epochData3D, (1,-1,numSamples))[0,:,:];
    #epochDataMerged = epochDataMerged.round(decimals=3);

    # Analyse epoch data list
    numRows = epochDataMerged.shape[0];
    subsetIdx = epochData.selection;

    # Channel info
    channels = epochData.picks+1;
    channels_repeat = np.resize(channels, numRows); # Ensure channels loops for all the rows

    # Onset sample
    onsetSmpl = np.round(epochData.annotations.onset * epochData.info["sfreq"])[subsetIdx];
    onsetSmpl_repeat = np.repeat(onsetSmpl, len(channels)); # Each onset is for an epoch, which spans all channels

    # Condition
    # OLD METHOD DOESN'T WORK BECAUSE THIS FUNCTIONS COLLECTS ALL ANNOTATIONS THAT OCCUR DURING THE EPOCH (IF OVERLAP), THEREFORE
    # IT COULD USE AN INCORRECT EPOCH
    # condition = np.array([ep[0][2] for ep in epochData.get_annotations_per_epoch()]);
    inv_events_dict = {v: k for k, v in eventDict.items()};
    condition = np.array([inv_events_dict[e_tag[2]] for e_tag in events]);
    condition = condition[subsetIdx];
    condition_repeat = np.repeat(condition, len(channels)); # Each condition is for an epoch, where an epoch spans all channels. Therefore repeat for all channels

    # Get chromophore & ROI
    roi = getROI(channels_repeat, 'EG_infant_8x8');
    chromophore = getChromophore(channels_repeat, 'EG_infant_8x8');

    # ThisTokenCount: Specific for each trigger
    # Repeats for all of the channels
    # All token count: General count
    # NOTE: Assumes conditions (triggers) are in sequential order. Should sequentially order them first
    allTokenCount = np.arange(1,len(condition)+1);
    allTokenCount_repeat = np.repeat(allTokenCount, len(channels)); # Repeat for all channels

    thisTokenCountDict = {};
    thisTokenCount = [];

    speechType = [];
    for trig in condition:
        if (trig in thisTokenCountDict):
            thisTokenCountDict[trig] = thisTokenCountDict[trig] + 1; # Increment count
        else:
            thisTokenCountDict[trig] = 1;

        # Add token count - so we know how many trials have been presented
        thisTokenCount.append(thisTokenCountDict[trig]);

        # Determine the which test was done (called speech type in the dataframe)
        spTy = trig[-3:];
        if (spTy == 'Det'):
            speechType.append("Detection");
        elif (spTy == 'Nov'):
            speechType.append("Discrimination");
        else:
            speechType.append("Unknown");

    thisTokenCount_repeat = np.repeat(thisTokenCount, len(channels)); # Repeat for all channels
    speechType_repeat = np.repeat(speechType, len(channels)); # Repeat for all channels

    # Determine Epoch/Timevec formatting - we are always outputting the dataframe in csv format

    numSampl = epochDataMerged.shape[1];
    epochDf = pd.DataFrame(epochDataMerged, columns=['Epoch_'+str(num) for num in np.arange(1, numSampl+1)]);
    timeNpArr = np.tile(epochData.times, (numRows, 1));
    timeSampl = timeNpArr.shape[1];
    timeDf = pd.DataFrame(timeNpArr, columns=['TimeVec_'+str(num) for num in np.arange(1, timeSampl+1)]);
    allData = {
            "Condition": condition_repeat,
            "Channel": channels_repeat,
            "ThisTokenCount": thisTokenCount_repeat,
            "AllTokenCount": allTokenCount_repeat,
            "SpeechType": speechType_repeat,
            # "ChannelSCI": sci_repeat,
            "OnsetSample": onsetSmpl_repeat,
            "ROI": roi,
            "Chromophore": chromophore,
            "SubID": 'testSubID'
        }

    epochDataFrame = pd.DataFrame(allData);
    epochDataFrame = pd.concat([epochDf, timeDf, epochDataFrame], axis=1);


    protocol = 'data_qual_test'
    dataType = 'Real data'
    
    epochDataFrame = epochDataFrame.assign(Protocol=protocol);
    epochDataFrame = epochDataFrame.assign(ControlType=dataType);
    
    #data quality measure(s)
    
    event_samples = events[:,0]
    
    for this_quality_measure in data_quality_info:
        
        this_quality_measure_vals = []
        
        for i in epochDataFrame['OnsetSample'].unique().tolist(): 
            #Using the output dataframe to define where the triggers are since sometimes the last trig is dropped by the epoching function
            #Find the closest window-start that corresponds to this event
            time_idx = np.abs(np.array(window_parameters['win_start_indices'])-i).argmin()
            
            for ch_idx, ch in enumerate(data.ch_names):
                
                this_quality_measure_vals.append(data_quality_info[this_quality_measure][ch_idx,time_idx])

        epochDataFrame[this_quality_measure] = this_quality_measure_vals
        
    #Add on a row for min/max value in epoch
    epochs_only = 1e6 * epochDataFrame.filter(like='Epoch').values
    epochDataFrame['MaxVal'] = np.max(np.abs(epochs_only),axis=1)

    return epochDataFrame, events


#OUTDATED VISUALISATION FUNCTION
def visualise_quality_measure(processed_data, window_parameters, data_quality_info):
    
    def normalise_array(array_in):
        array_out = array_in/(np.max(array_in)-np.min(array_in));
        return array_out
    
    #For each quality measure, plot out each HbO channel and their corresponding data quality measure over time
    
    continuous_data = processed_data.get_data();
    plot_scale = np.std(continuous_data) #(np.max(continuous_data) - np.min(continuous_data)) / 20
    
    for i in quality_measures:
        plt.figure()
        plt.title(i)
        this_measure = normalise_array(quality_measures[i])
        for ch_idx in range(0,len(processed_data.ch_names),2):
            
            plt.plot(processed_data.times, plot_scale*ch_idx + continuous_data[ch_idx,:],linewidth=0.3,color='black')
            plt.plot((np.array(window_parameters['win_start_indices'])/processed_data.info['sfreq']),
                     plot_scale*ch_idx + this_measure[ch_idx,:]*plot_scale,
                     linewidth=0.5, color = 'blue')

#OUTDATED VISUALISATION FUNCTION  
def visualise_rejections(processed_data, epochDataFrame, event_times, rejection_thresholds):
    
    continuous_data = processed_data.get_data();
    plot_scale = np.std(continuous_data) #(np.max(continuous_data) - np.min(continuous_data)) / 20
    time_vec = epochs.iloc[0].filter(like='Time').tolist()
    
    plt.figure()
    
    for ch_idx in range(0,len(processed_data.ch_names),2):
        
        plt.plot(processed_data.times, plot_scale*ch_idx + continuous_data[ch_idx,:],linewidth=0.1,color='black')
        
        for evt_idx, evt_time in enumerate(event_times):
            
            filtered_df = epochDataFrame[
                (epochDataFrame['Channel'] == (ch_idx+1)) &
                (epochDataFrame['Chromophore'] == 'HbO') & 
                (epochDataFrame['OnsetSample'] == evt_time)]
            
            epoch_color = 'blue'
            for col, thresh in rejection_thresholds.items():
                if filtered_df[col].iloc[0] <= thresh:
                    epoch_color = 'red'
                    
            hbo_subset_average = filtered_df.filter(like='Epoch').values.flatten()
            
            plt.plot(time_vec + evt_time/processed_data.info['sfreq'], plot_scale*ch_idx + hbo_subset_average,
                     linewidth = 1, color=epoch_color)
    
    
    
    
    
'''
Example usage
'''
    
if (__name__ == '__main__'):
    
    #Sample usage of this script
    
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)
    root.update()
    fname = filedialog.askdirectory(initialdir = r"O:\HUM\Projects\fNIRS\EarGenie data\EarGenie Speech Module Development\S012.F and S013.F SpeechDetect(interleave) & SpeechDiscrim(staircase)_VaryingIntensity_Oct2022\04 Raw Data")
    
    # raw_data = mne.io.read_raw_nirx(fpath);
    
        ###REGULAR PREPROCESSING###
    # od_data = mne.preprocessing.nirs.optical_density(raw_data)
    
    #Instead of processing a real dataset, simulate some responses and add it to the baseline segment of a file
    target_amp = 20
    # fname = r'C:\Users\maod\Downloads\sample_data\C212.H103.F.HI\2025-05-12_001' #This is the super noisy dataset
    # fname = r'C:\Users\maod\Downloads\sample_data\C168.S012.F\2024-07-05_001' #This is the clean dataset

    match = re.search(r'C\d{3}', fname)
    subID = match.group()

    epochs_ex, od_rest, od_ex = simulate_exData(fname,target_amp) 
    
    od_data = od_ex.copy();
    original_od_data = od_data.copy()

    #bandpass filter
    lower_cutoff = 0.01; # Or the high pass cutoff
    upper_cutoff = 0.25; # Or the low pass cutoff

    # band pass filter the optical density
    data_filtered = od_data.filter(l_freq=lower_cutoff,
                        h_freq=upper_cutoff,
                        method='iir',
                        iir_params = dict(order=8,
                                          ftype='butter',
                                          output='sos'));

    
    #haemodynamic conversion
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(data_filtered, ppf=0.1);
        
        ###DATA QUALITY CALCULATIONS###

    # Define sliding window
    window_length = 30.0
    window_step = 10.0
    window_parameters = def_window(original_od_data, window_length=window_length, window_step=window_step)
    
    # Calculate metrics per window

    quality_measures = {'PSP':      peak_power(original_od_data, window_parameters),
                        'SCI':      sci(original_od_data, window_parameters),
                        'Corr':     avg_correlation(original_od_data, window_parameters),
                        'Bounce':   induce_bounce(original_od_data, window_parameters)}
    
    #Epoching, with quality measures included per epoch
    epochs, events = data_to_epoch_table(raw_haemo, window_parameters, quality_measures)

    #Rejection based on certain thresholds
    # rejection_thresholds = {
    #     'PSP':0.1,
    #     'SCI':0.6,
    #     'Corr':0.5}
    
    #Visualise each rejection measure
    # visualise_quality_measure(raw_haemo, window_parameters, quality_measures)
    
    #reject and plot
    # visualise_rejections(processed_data = raw_haemo, 
    #                      epochDataFrame = epochs, 
    #                      event_times = events[:,0], 
    #                      rejection_thresholds=rejection_thresholds
    #                      )


    #temp code: Get the ideal response size (needed as the simulation doesn't produce an exact response)
    od_pure = od_ex.copy() #temporary version of dataset with just the response and no noise
    pure_resp_data = od_ex.get_data() - od_rest.get_data()
    od_pure._data = pure_resp_data
    od_pure_filtered = od_pure.filter(l_freq=lower_cutoff,
                        h_freq=upper_cutoff,
                        method='iir',
                        iir_params = dict(order=8,
                                          ftype='butter',
                                          output='sos'));
    od_pure_haemo = mne.preprocessing.nirs.beer_lambert_law(od_pure_filtered, ppf=0.1);
    
    pure_epochs, _ = data_to_epoch_table(od_pure_haemo, window_parameters, quality_measures)
    pure_hbo = pure_epochs[(pure_epochs['Chromophore'] == 'HbO') & (pure_epochs['Condition'] == '8.0')] #Can filter here for specific channels/ROI as well
    pure_hbo_epochs = 1e6 * pure_hbo.filter(like='Epoch').values
    pure_hbo_avg = np.mean(pure_hbo_epochs,0)

    #Plot the averaged epoch
    t_vec = epochs.filter(like='Time').values[0,:]
    epochs = epochs[(epochs['Chromophore'] == 'HbO') & (epochs['Condition'] == '8.0')]
    a = 1e6 * epochs.filter(like='Epoch').values
    this_mean = np.mean(a,0)
    this_stdev = np.std(a,0)
    
    #%%
    
    #show simulated epochs
    fig = plt.figure()
    plt.rcParams.update({'font.size': 16}) 
    
    fig.suptitle(subID)
    
    ax1 = plt.subplot(2,2,1)
    ax1.plot(t_vec, np.mean(a,0), label = 'Full sim (' + str(len(epochs)) + ') epochs')
    ax1.fill_between(t_vec, this_mean-this_stdev, this_mean+this_stdev, alpha=0.1)
    ax1.plot(t_vec,np.mean(pure_hbo_epochs,0), label = 'Sim without noise')
    ax1.fill_between(t_vec, np.mean(pure_hbo_epochs,0)-np.std(pure_hbo_epochs,0), np.mean(pure_hbo_epochs,0)+np.std(pure_hbo_epochs,0), alpha=0.2)
    ax1.plot(t_vec,[target_amp]*len(t_vec), label = 'Target sim amplitude')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('HbO (mM)')
    ax1.legend()

    #show raw data
    ax2 = plt.subplot(1,2,2)
    continuous_data = od_data.get_data();
    plot_scale = np.std(continuous_data)
    
    for ch_idx in range(0,len(od_data.ch_names),2):
        
        ax2.plot(od_data.times, plot_scale*ch_idx + continuous_data[ch_idx,:],linewidth=0.5,color='black')


    #Try different thresholds for rejecting the epochs and show change in data quality
    
    quality_thresholds = {'PSP':    np.arange(0, 3.05, 0.1),
                          'MaxVal': np.arange(50,200,10),
                          'Corr':   np.arange(0,1.05,0.1),
                          'Bounce': np.arange(0,1.05,0.1),
                          'SCI':    np.arange(0,1.05,0.1)
                          }

    
    ax3 = plt.subplot(2,2,3)
    
    for this_quality_measure in quality_thresholds:
        
        quality_threshold_range = quality_thresholds[this_quality_measure]
        
        result_rmse = np.zeros(len(quality_threshold_range))
        result_epochs_kept = np.zeros(len(quality_threshold_range))
        
        for idx, this_threshold in enumerate(quality_threshold_range):
            this_df = epochs[(epochs[this_quality_measure] > this_threshold)] #Can filter here for specific channels/ROI as well
            this_epochs = 1e6 * this_df.filter(like='Epoch').values
            result_rmse[idx] = rms(pure_hbo_avg - np.mean(this_epochs,0))
            result_epochs_kept[idx] = len(this_epochs)/len(epochs)
            
        ax3.plot(result_epochs_kept, result_rmse, label=this_quality_measure, marker = 'o')
        
    ax3.set_xlabel('Proportion of epochs kept')
    ax3.set_ylabel('RMS error')
    ax3.legend()
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
    plt.pause(0.1)
    fig.canvas.draw()
    fig.savefig(subID+'.png', format='png',bbox_inches='tight')
    
    
    
    
                
#perhaps rejection if PSP/FC is bad and SCI is good; otherwise leave in for correction (SCI bad = no large artefacts?)


