import os 
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"


import pandas as pd
import numpy as np
import os
import glob

from fnirs_preproc.src.nirs_read_raw import read_nirx

from profiler import single_run_profile, save_corr_vals
from pathlib import Path

from profiler import save_profile, runLiveSim
import traceback
from datetime import datetime

import json
import requests
HERE = Path(__file__).resolve().parent

def load_online_json(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Failed to download file: {response.status_code}")
            return {}
    except Exception as e:
        traceback.print_exc()
        return {}


def load_local_json(path):
    try:
        with open(path, 'r') as file:
            return json.load(file)

    except Exception as e:
        traceback.print_exc()
        return {}


def renameRecordingFolders(search_base_folder): 
    for dir_path,dirs,files in os.walk(search_base_folder):
        valid_nirs_folder_count = 0;
        for fname in os.listdir(dir_path):
            if (fname.endswith('.evt') or 
                fname.endswith('.wl1') or
                fname.endswith('.wl2')): # Test if contains wavelength file
                valid_nirs_folder_count += 1;
                # Run preprocessing on this folder (dir_path), and use dir_name as subID
        
        if (valid_nirs_folder_count >= 3):
            # All requirements were met
            inner_folder = os.path.basename(dir_path);
            subID = os.path.basename(os.path.dirname(dir_path));
            src = f"{subID}.{inner_folder}"
            prot = src.split('.')[1]
            
            if (inner_folder.split('.')[0] != prot):
            
                # Protocol not been appended
                
                # New folder name with prot appended to the front
                new_inner_folder = prot + '.' + inner_folder
    
                # Full path to the new folder
                new_dir_path = os.path.join(os.path.dirname(dir_path), new_inner_folder)
    
                # Rename the folder
                os.rename(dir_path, new_dir_path)



def search_for_not_run(search_base_folder, output_folder):
    
    # Read excel filenames in folder
    # Use glob to match the pattern '**/*.xlsx', looking in subdirectories
    files = glob.glob(os.path.join(output_folder, "**/*.xlsx"), recursive=True)
    
    # Get file names without extension and store them in an array
    existing_subjects = [os.path.splitext(os.path.basename(file))[0] for file in files]
    existing_subjects = list(set( existing_subjects ))
    print("========= SUBJECTS RUN ==========\n" + '\n'.join(existing_subjects))
    print("\n\n")
    
    src_not_run = []
    runArr = []
    
    for dir_path,dirs,files in os.walk(search_base_folder):
        valid_nirs_folder_count = 0;
        pkl_file = None
        for fname in os.listdir(dir_path):
            if (fname.endswith('.evt') or 
                fname.endswith('.wl1') or
                fname.endswith('.wl2')): # Test if contains wavelength file
                valid_nirs_folder_count += 1;
                # Run preprocessing on this folder (dir_path), and use dir_name as subID
            
            if (fname.endswith('.pkl')):
                pkl_file = os.path.join(dir_path, fname)
            
        if ((valid_nirs_folder_count >= 3) or (pkl_file is not None)):
            # All requirements were met
            inner_folder = os.path.basename(dir_path);
            subID = os.path.basename(os.path.dirname(dir_path));
            src = f"{subID}.{inner_folder}"


            
            split_inner = inner_folder.split('.')
            if (len(split_inner) > 1):
                # IF contains more than 1 segment, then assume protocol embedded in recording label
                prot = inner_folder.split('.')[1]
            else:
                # Take from outer folder name (LEGACY => should not be used)
                print(f"======== Used legacy protocol extraction for recording {inner_folder} =========")
                prot = src.split('.')[1]
            
            stripped_path = dir_path.replace(search_base_folder, '')  # Strip the base folder
            stripped_path = stripped_path.lstrip(os.sep)  # Ensure no leading path separator
            save_file_name = subID + "-" + prot # Replace path separators with underscore
            
            if (save_file_name not in existing_subjects):
                # Found a subject to run
                src_not_run.append(src)
                
                if (valid_nirs_folder_count >= 3):
                    runArr.append( {
                            "src": src,
                            "dir_path": dir_path,
                            "save_path": save_file_name,
                            "prot": prot
                        } )
                    
                elif (pkl_file is not None):
                    runArr.append( {
                            "src": src,
                            "dir_path": pkl_file,
                            "save_path": save_file_name,
                            "prot": prot
                        } )
                    
                    
                
    
    print("========= SUBJECTS NOT RUN ==========\n" + '\n'.join(src_not_run))
    
    return runArr

def swapColumns(raw):
    block2_start = raw['trigger_samples'][9]
    block2_end = raw['trigger_samples'][15]
    block1_end = raw['trigger_samples'][8]

        # Get Block 1 (first p columns) and Block 2 (r columns starting at block2_start)
    block1 = raw['data'][:, 1:block1_end]
    block2 = raw['data'][:, block2_start:block2_end]

    # Swap the blocks
    total_columns_block2 = block2.shape[1]
    total_columns_block1 = block1.shape[1]


    raw['data'][:, 1:total_columns_block2+1] = block2
    raw['data'][:, block2_start:block2_start + total_columns_block1] = block1

    return raw

            
            
def run_all(search_base_folder, output_folder, protocols):

    # Make output dirs
    
    withPlotsDir = os.path.join(output_folder, "WithPlots")
    if (not os.path.isdir(withPlotsDir)):
        os.mkdir(withPlotsDir)
    
    withoutPlotsDir = os.path.join(output_folder, "NoPlots")
    if (not os.path.isdir(withoutPlotsDir)):
        os.mkdir(withoutPlotsDir)
        
    # Find subjects that have/haven't run
    runArr = search_for_not_run(search_base_folder, output_folder)
    
    failed_runs = []
    run_id = 0

    
    for runObj in runArr:
        try:
            
            
            src = runObj['src']
            prot = runObj['prot']
            dir_path = runObj['dir_path']
            save_file_name = runObj['save_path']

            subIDList=[src]
            df_file=pd.DataFrame([subIDList])
            df_file[0].to_csv("subject_id.csv")
            
            print(f"=================== Reading Raw {src} ========================")
            raw = read_nirx(dir_path, protocols[prot])
            #raw = swapColumns(raw) - This method was only created to see the impact of changing the silencebaseline to the 2nd half of the silence block instead of the 1st. REsults are stored in the folder: "C:\Users\BalasuG\OneDrive - The Bionics Institute of Australia\Documents\EarGenie\Experiments\SilenceBaseline_InterspersedSilence_Investigation\0dB_FPTestValidation_flipped"
            
            print(f"=================== Starting Processing {src} ========================")
            
            runObj['finished'] = False
            
            # Run both sides
            sides = ['L', 'R']
    
            side_dfs = []
              
            perf_df = runLiveSim(raw, save_failed_runs=False)
            side_dfs.append(perf_df)
    
            perf_df = pd.concat(side_dfs)
            perf_df["SubjectID"] = src
            perf_df["Protocol"] = prot
                
            #fileName_plots = os.path.join(withPlotsDir, f"{save_file_name}.xlsx")
            #save_profile(perf_df, fileName_plots, plot=True, raw=raw, autoOpen=False)
            
            
            fileName_noplots = os.path.join(withoutPlotsDir, f"{save_file_name}.xlsx")
            save_corr_vals(perf_df, fileName_noplots, plot=False, raw=raw, autoOpen=False)
            runObj['finished'] = True
            
            print(f"=================== Finished Processing {src} ========================")
            
            
        except Exception as e:
            print(f"=================== !ERROR! Processing {src} ========================")
            traceback.print_exc()
            failed_runs.append(src)
            
            run_id += 1  # increment the run ID
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # get current date and time in the format YYYY-MM-DD HH:MM:SS
            with open('errors.txt', 'a') as f:
                f.write(f"=================== !ERROR! Processing {src} ========================\n")
                f.write(f"Run ID: {run_id}, Date: {current_date}, Filename: {dir_path}\n")  # write the run ID and date to the file
                f.write(f"{traceback.format_exc()}\n")
                failed_runs.append(src)
    
    
    print("========= FAILED SUBJECTS ==========\n" + '\n'.join(failed_runs))
        
    
            
if (__name__ == '__main__'):
    
    link = ""

    filename =      HERE / "protocols.json"
    download_link = link + "?download=1"
   # protocols = load_online_json(download_link) # place the protocols.json file in the base folder in a remote link. 
    protocols = load_local_json(filename) # place the protocols.json file in the base folder in     a remote link. 


    print(protocols)
    #output_folder = r'C:\\Users\\BalasuG\\OneDrive - The Bionics Institute of Australia\\Documents\\EarGenie\\Experiments\\Alg_V2_Integration' # empty to begin with
    #output_folder = r'C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\c184_results' # empty to begin with
    output_folder = r'C:\\Users\\GBalasubramanian\\OneDrive - The Bionics Institute of Australia\\Documents\\FMRI_FNIRS_Work\\Experiments\\fnirsCorrelationValues\\output'


    #output_folder = r"C:\\Users\\BalasuG\\Downloads\\0dB Test Results Discrim\\closerinit_fptest_adjusted_pt5_sigmaf_dynamic_merge_whitepaper\\nh\\test_early_stop"
    #search_base_folder = r'C:\Users\BalasuG\Downloads\DetTest_Aug23\Detection_2' # replace with folder location of nirx data. 
    #search_base_folder = r'C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\C143.S012\\EarGenie\\'
    #search_base_folder = r'C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\C184_RE.S016.F.AN'
    #search_base_folder = r"C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\May 2024\\NH\\Detection_3"
    #search_base_folder = r'C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\May 2024\\NH'
    #search_base_folder = r'C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\sanity_check\\'
    #search_base_folder = r'C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\May 2024\\NH\\Detection_1\\2_C077.S008.modified.yessilence.yeshabituation'
    search_base_folder = r'C:\\Users\\GBalasubramanian\\Downloads\\cogT_fnirs'

    
    #search_base_folder = r'C:\\Users\\BalasuG\\Downloads\\FNIRS Testing\\awake data\\rerun'
    run_all(search_base_folder, output_folder, protocols)