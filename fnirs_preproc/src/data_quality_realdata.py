# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:24:30 2025

@author: dmao
"""

import data_quality_2

import numpy as np
import pandas as pd 

import mne

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the root window
root.attributes("-topmost", True)
root.update()
fpath = filedialog.askdirectory(initialdir = r"O:\HUM\Projects\fNIRS\EarGenie DSP figs and notes\Gautam\fnirs testing\Julia\Detection_1\10_C054.S008.F_modified_checkage.yessilence.yeshabituation")

raw_data = mne.io.read_raw_nirx(fpath);
import os
print("You selected:", fpath)
print("Files in that folder:", os.listdir(fpath))

    ###REGULAR PREPROCESSING###
# 1. Optic density
od_data = mne.preprocessing.nirs.optical_density(raw_data)

original_od_data = od_data.copy()
# 2. TDDR

raw_tddr = mne.preprocessing.nirs.temporal_derivative_distribution_repair(od_data);
#raw_tddr = od_data.copy()

#bandpass filter
lower_cutoff = 0.01; # Or the high pass cutoff
upper_cutoff = 0.25; # Or the low pass cutoff

# band pass filter the optical density

#3. Band pass Filtering 
data_filtered = raw_tddr.filter(l_freq=lower_cutoff,
                    h_freq=upper_cutoff,
                    method='iir',
                    iir_params = dict(order=8,
                                      ftype='butter',
                                      output='sos'));


#4. HbO HbR aemodynamic conversion


raw_haemo = mne.preprocessing.nirs.beer_lambert_law(data_filtered, ppf=0.1);
    
    ###DATA QUALITY CALCULATIONS###

# Define sliding window
window_length = 30.0
window_step = 10.0
window_parameters = data_quality_2.def_window(original_od_data, window_length=window_length, window_step=window_step)

# Calculate metrics per window

quality_measures = {'PSP':      data_quality_2.peak_power(original_od_data, window_parameters),
                    'SCI':      data_quality_2.sci(original_od_data, window_parameters),
                    'Corr':     data_quality_2.avg_correlation(original_od_data, window_parameters),
                    'Bounce':   data_quality_2.induce_bounce(original_od_data, window_parameters)}

#Epoching, with quality measures included per epoch
epochs, events = data_quality_2.data_to_epoch_table(raw_haemo, window_parameters, quality_measures)

epochs.loc[:, epochs.columns.str.startswith('Epoch')] *= 1e6
epochs.to_csv('test_epochs.csv')
