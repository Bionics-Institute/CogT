# If called from same script or a script in the same directory
if __name__=='__main__' or __name__=='fnirs_preprocessing_functions':
    from fnirs_preprocessing_functions import fnirs_preprocessing, spaced_random_samples
    from nirs_read_raw import read_nirx, concat_raw_outputs
else:
    # Remove dot if calling file from another folder    
    from .fnirs_preprocessing_functions import fnirs_preprocessing, spaced_random_samples
    from .nirs_read_raw import read_nirx, concat_raw_outputs

import numpy as np
import pandas as pd
import os 


def preprocess_nirx(parent_folder, protocols):
    
    
    # This will hold all the preprocessed outputs
    all_outputs = {}

    # Walk through all directories
    for dir_path,dirs,files in os.walk(parent_folder):
        valid_nirs_folder_count = 0;
        for fname in os.listdir(dir_path):
            if (fname.endswith('.evt') or 
                fname.endswith('.wl1') or
                fname.endswith('.wl2')): # Test if contains wavelength file
                valid_nirs_folder_count += 1;
                # Run preprocessing on this folder (dir_path), and use dir_name as subID
        
        if (valid_nirs_folder_count >= 3):
            # All requirements were met
            date = os.path.basename(dir_path);
            subID = os.path.basename(os.path.dirname(dir_path));
            
            try:   
                protStr = subID.split('.')[1]
                prot = protocols[protStr]
                print(f"===== Successfully loaded protocol {protStr} =======")
            except:
                prot = {}
                print(f"===== Failed to load protocol, using blank =======")
            
            output = read_nirx(dir_path, prot)
            print(f"Reading {subID}...")
            if (subID in all_outputs):
                all_outputs[subID].append(output)
            else:
                all_outputs[subID] = [output]

    # concatenate the raw output using concat_raw_outputs and run preproc
    
    dfs = []
    for subID in all_outputs.keys():
        raw = concat_raw_outputs(all_outputs[subID])
        print(f"Preprocessing {subID}...")
        outdf = preprocess_raw(raw)
        outdf["SubID"] = subID
        dfs.append(outdf)
    
    final_output = pd.concat(dfs)
    
    return final_output


def preprocess_raw(raw_output):
    
    # Input into preprocessing function 
    out_df, _ = fnirs_preprocessing(raw_output['data'], 
                                    raw_output['sfreq'], 
                                    raw_output['trigger_samples'], 
                                    raw_output['trigger_labels'],
                                    raw_output['wavelengths'], 
                                    raw_output['ch_labels'], 
                                    raw_output['distances'], 
                                    raw_output['roi'])
        
    
    # Generate controls
    min_dist = int(20 * raw_output['sfreq']); # i.e. 20 seconds minimum distance
    min_smpl = 0;
    max_smpl = raw_output['data'].shape[1] - int(30  * raw_output['sfreq']); # Leave 30 seconds at the end for PSR
    num_smpl = len(raw_output['trigger_samples']);
    
    ctrl_trig_samples = spaced_random_samples(min_smpl, max_smpl, num_smpl, min_dist);
    ctrl_trig_labels = np.array(["Random"]*num_smpl)
    
    
    ctrl_df, _ = fnirs_preprocessing(raw_output['data'], 
                                    raw_output['sfreq'], 
                                    ctrl_trig_samples, 
                                    ctrl_trig_labels,
                                    raw_output['wavelengths'], 
                                    raw_output['ch_labels'], 
                                    raw_output['distances'], 
                                    raw_output['roi'])
    
    # Combine
    out = pd.concat([out_df, ctrl_df])
    
    return out
     

if __name__=='__main__':
    from protocols import default as protocol


    nirx_str = r"C:\Users\SanjayanA\OneDrive - The Bionics Institute of Australia\Documents\Data"
    
    
    outdf = preprocess_nirx(nirx_str, protocol)
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    outdf.to_csv("data/preprocessing.csv", index=False);
    