
import numpy as np
import time
import pandas as pd
import copy
from subprocess import Popen
import sys, os
import matplotlib.pyplot as plt
import xlsxwriter
from datetime import datetime

import requests
import traceback
import json
from runAlg import main

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


def mask_side(raw_i, side_to_use="L"):
    
    raw = raw_i.copy()
    
    if (side_to_use == "L"):
        mask = (raw['roi'] == 'LF') | (raw['roi'] == 'LT') | (raw['roi'] == 'Cross-L')
    else:
        mask = (raw['roi'] == 'RF') | (raw['roi'] == 'RT') | (raw['roi'] == 'Cross-R')
    
    # Mapping dictionary for conversion
    #mapping = {'LF': 'Frontal', 'RF': 'Frontal', 'LT': 'Temporal', 'RT': 'Temporal', 'Cross-L': 'Cross', 'Cross-R': 'Cross'}

    mapping = {'LF': 'Prefrontal', 'RF': 'Prefrontal', 'LT': 'Temporal', 'RT': 'Temporal', 'Cross-L': 'Cross', 'Cross-R': 'Cross'}
    
    
    # Perform the conversion using vectorized operations
    raw['roi'] = np.array([mapping.get(item, item) for item in raw['roi']])
    raw['data'] = raw['data'][mask, :]
    raw['wavelengths'] = raw['wavelengths'][mask]
    raw['ch_labels'] = raw['ch_labels'][mask]
    raw['distances'] = raw['distances'][mask]
    raw['roi'] = raw['roi'][mask]
    
    return raw




def single_run_profile(raw, epoch_bounds=[-3,27], save_failed_runs=True, fb_prms = None,
                       print_func = print):

    if (fb_prms is None):
        fb_prms = {
            "run_number": 1,
            "perf_df_list": [],
            "last_run_time": 0,
            "last_start_sampl": 0,
            "regionScoreListTemporal": {},
            "regionScoreListFrontal": {},
            "isStopObj": {},
            "finished": False
            }

    post_sampl = np.ceil(epoch_bounds[1] * raw['sfreq']) # Samples to wait until run trigger
    
    trig_psr_samples = raw['trigger_samples'] + post_sampl # Samples at which trigger PSRs are available
    
    alg_end_sampl = fb_prms["last_run_time"] * raw['sfreq'] + fb_prms["last_start_sampl"]; # Sample we are at after running algorithm
    
    
    # Determine if any trigger psrs were ready
    ''' 
    trig_proc_idx = np.where((trig_psr_samples > fb_prms["last_start_sampl"]) & 
                       (trig_psr_samples <= alg_end_sampl)); # Get indices which are greater than the last run sample, and smaller/eq than alg

    trig_proc_idx = trig_proc_idx[0].tolist();
    '''
    trig_proc_idx=[] # Added to ensure each trial gets processed in the profiler. 
    if len(trig_proc_idx) > 0:
        # Collected triggers during alg run time, thus start running alg straight away
        fb_prms["last_start_sampl"] = alg_end_sampl; 
        
        new_trigger_labels = raw['trigger_labels'][trig_proc_idx];
        
        max_trig_psr_sample = int(max(trig_psr_samples[trig_proc_idx]))
    
    else:
        # No triggers collected during alg run time, so choose the next one
        t_psrs_left = trig_psr_samples[trig_psr_samples > fb_prms["last_start_sampl"]]
        
        if (len(t_psrs_left) <= 0):
            # We have no more triggers to process 
            fb_prms["finished"] = True
            return fb_prms
        
        # Choose first trigger psr after algorithm started previously - if running in batch change no of triggers to point to the end
        fb_prms["last_start_sampl"] = t_psrs_left[0] 
        
        max_trig_psr_sample = int(fb_prms["last_start_sampl"])
        
        new_trigger_labels = raw['trigger_labels'][trig_psr_samples == fb_prms["last_start_sampl"]]
    
    # Data to pass
    collected_data = copy.deepcopy(raw)
   
    # Remove parts of data
    trig_mask = trig_psr_samples <= max_trig_psr_sample
    
    collected_data['trigger_labels'] = collected_data['trigger_labels'][trig_mask]
    collected_data['trigger_samples'] = collected_data['trigger_samples'][trig_mask]
    
    collected_data['data'] = collected_data['data'][:, :max_trig_psr_sample+1]

    
    # Pass data to algorithm with performance test
    s_alg_time = time.perf_counter()
    
    output_df = None
    
    try:

        output_df  =main(raw['data'], 
                collected_data['sfreq'], 
                collected_data['trigger_labels'], 
                collected_data['trigger_samples'], 
                collected_data['distances'], 
                collected_data['wavelengths'], 
                collected_data['ch_labels'])
        


    
    except:
        traceback.print_exc()
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # get current date and time in the format YYYY-MM-DD HH:MM:SS
        with open('interface_errors.txt', 'a') as f:
            f.write(f"Token: {new_trigger_labels}, Run ID: {fb_prms['run_number']}, Date: {current_date}\n")  # write the run ID and date to the file
            f.write(f"{traceback.format_exc()}\n")

    e_alg_time = time.perf_counter()
    
    
    fb_prms["last_run_time"] = e_alg_time - s_alg_time # + 50 # <-- artifical delay
    
    # Save to df if df not none
    if (output_df is not None):
        output_df.insert(0, 'Run', fb_prms["run_number"])
        output_df.insert(1, '@StartTime', fb_prms["last_start_sampl"] / raw['sfreq'])
        output_df.insert(2, 'RunTime', fb_prms["last_run_time"])
        
       
    
        fb_prms["perf_df_list"].append(output_df)
    
    else:
        if (save_failed_runs):
            debug_df = pd.DataFrame({'Token': new_trigger_labels})
            debug_df.insert(0, 'Run', fb_prms["run_number"])
            debug_df.insert(1, '@StartTime', fb_prms["last_start_sampl"] / raw['sfreq'])
            debug_df.insert(2, 'RunTime', fb_prms["last_run_time"])
            
            fb_prms["perf_df_list"].append(debug_df)
            
            
            
        
    txt = f"""<br/>
    -------------------------------<br/>
    Run Number: {fb_prms["run_number"]}<br/>
    Run Success: <font color='{"green" if output_df is not None else "red"}'>{output_df is not None}</font><br/>
    Token List: {', '.join(map(str, new_trigger_labels))}<br/>
    Algorithm Start Time: {round(fb_prms["last_start_sampl"] / raw['sfreq'], 4)} s <br/>
    Algorithm Run Time: {round(fb_prms["last_run_time"], 4)} s <br/>
    -------------------------------<br/>
    """
    print_func(txt)
    
    
    fb_prms["run_number"] += 1
    
    return fb_prms
    
    


def get_cell_value(value):
    if isinstance(value, np.ndarray):
        if pd.isnull(value).any():
            return ""
        else:
            return str(value)
    elif pd.isnull(value):
        return ""
    else:
        return str(value)



def rgb_to_hex(r,g,b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def save_corr_vals(perf_df, save_file_path, plot=False, raw=None, epoch_bounds=[0,18], autoOpen = True, subID = None):
    if (perf_df is None):
        print("Nothing to output. Exiting...")
        return
    
    
    # Assign subID if present
    if (subID is not None):
        perf_df["SubjectID"] = subID
        
    # Round to 4 decimals max
    perf_df = perf_df.round(4)
        
    # Pre org
    #perf_df = perf_df.sort_values(["NoofTrials", "Token", "Side", "Region", "TestType"], ascending=[True, True, True, True, False])

    
    if (plot==True):
        # Append the new directory to your file path
        p_folder = os.path.dirname(save_file_path)
        plot_dir = os.path.join(p_folder, "plots")
        
        # Create the new directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    workbook = xlsxwriter.Workbook(save_file_path)
    worksheet = workbook.add_worksheet()
    
    header_format = workbook.add_format({'bg_color': 'black', 'font_color': 'white'})
    border_format_top = workbook.add_format({'top': 1, 'top_color': 'black'})
    border_format_bot = workbook.add_format({'bottom': 1, 'bottom_color': 'black'})
    red_format = workbook.add_format({'bg_color': rgb_to_hex(255,199,206), 'font_color': rgb_to_hex(156,0,6)})
    green_format = workbook.add_format({'bg_color': rgb_to_hex(198, 239, 206), 'font_color': rgb_to_hex(0,97,0)})
    yellow_format = workbook.add_format({'top': 1, 'top_color': 'black',
                                         'bg_color': rgb_to_hex(255,235,156), 
                                         'font_color': rgb_to_hex(156,101,0)})    
    blue_format = workbook.add_format({'bottom': 1, 'bottom_color': 'black',
                                        'bg_color': '#B7DEE8', 
                                        'font_color': '#244062'})
    c_hex = '#FECE00'
    psr_hex = "#8EAFD6"
    control_colour = workbook.add_format({'bg_color': c_hex})
    psr_colour = workbook.add_format({'bg_color': psr_hex})
    white_text = workbook.add_format({'font_color': rgb_to_hex(255,255,255)})

    
    # List of columns to exclude
    #exclude_columns = ["MeanEpochsHBO", "MeanEpochsHBR", "response"]
    out_df = perf_df
    
          
    # Write column headers to the Excel file
    for i, col in enumerate(out_df.columns):
        worksheet.write(0, i, col, header_format)
    
    min_y = np.inf
    max_y = -np.inf
    padding = (max_y - min_y) * 0.02
    min_y = min_y - padding
    max_y = max_y + padding

    data_row_index = 1
    i = 0
    while i < len(out_df):
        curr_row = out_df.iloc[i]
        curr_row_full = perf_df.iloc[i]
        if i + 1 < len(out_df) and ('@StartTime' in out_df.columns) and (out_df.iloc[i]['@StartTime'] == out_df.iloc[i+1]['@StartTime']):
            next_row = out_df.iloc[i+1]
            next_row_full = perf_df.iloc[i+1]
            for j in range(len(out_df.columns)):
                column_type = out_df.columns[j]
                f1 = None
                f2 = None
                if (pd.isna(curr_row['SubjectID']) or pd.isna(next_row['SubjectID'])):
                    if (column_type == 'Run'):
                        f1 = red_format
                        f2 = red_format
                else:
                    if (column_type == 'Run'):
                        f1 = green_format
                        f2 = green_format
                    elif (column_type == 'SubjectID' and curr_row[j] is not None and next_row[j] is not None):
                        f1 = yellow_format
                        f2 = blue_format
                        
                worksheet.write(data_row_index, j, get_cell_value(curr_row[j]), f1)
                worksheet.write(data_row_index + 1, j, get_cell_value(next_row[j]), f2)
    
            worksheet.set_row(data_row_index, None, border_format_top)
            worksheet.set_row(data_row_index + 1, None, border_format_bot)
            data_row_index += 2
            i += 2
        else:
            for j in range(len(out_df.columns)):
                worksheet.write(data_row_index, j, get_cell_value(curr_row[j]), red_format if j==0 else None)
            data_row_index += 1
            i += 1

       
    workbook.close()
    
    if (autoOpen):
        p = Popen(save_file_path, shell=True)


def save_profile(perf_df, save_file_path, plot=True, raw=None, epoch_bounds=[0,18], autoOpen = True, subID = None):
    
    if (perf_df is None):
        print("Nothing to output. Exiting...")
        return
    
    
    # Assign subID if present
    if (subID is not None):
        perf_df["SubjectID"] = subID
        
    # Round to 4 decimals max
    perf_df = perf_df.round(4)
        
    # Pre org
    #perf_df = perf_df.sort_values(["NoofTrials", "Token", "Side", "Region", "TestType"], ascending=[True, True, True, True, False])
    perf_df = perf_df.reset_index(drop=True)

    
    if (plot==True):
        # Append the new directory to your file path
        p_folder = os.path.dirname(save_file_path)
        plot_dir = os.path.join(p_folder, "plots")
        
        # Create the new directory if it doesn't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    workbook = xlsxwriter.Workbook(save_file_path)
    worksheet = workbook.add_worksheet()
    
    header_format = workbook.add_format({'bg_color': 'black', 'font_color': 'white'})
    border_format_top = workbook.add_format({'top': 1, 'top_color': 'black'})
    border_format_bot = workbook.add_format({'bottom': 1, 'bottom_color': 'black'})
    red_format = workbook.add_format({'bg_color': rgb_to_hex(255,199,206), 'font_color': rgb_to_hex(156,0,6)})
    green_format = workbook.add_format({'bg_color': rgb_to_hex(198, 239, 206), 'font_color': rgb_to_hex(0,97,0)})
    yellow_format = workbook.add_format({'top': 1, 'top_color': 'black',
                                         'bg_color': rgb_to_hex(255,235,156), 
                                         'font_color': rgb_to_hex(156,101,0)})    
    blue_format = workbook.add_format({'bottom': 1, 'bottom_color': 'black',
                                        'bg_color': '#B7DEE8', 
                                        'font_color': '#244062'})
    c_hex = '#FECE00'
    psr_hex = "#8EAFD6"
    control_colour = workbook.add_format({'bg_color': c_hex})
    psr_colour = workbook.add_format({'bg_color': psr_hex})
    white_text = workbook.add_format({'font_color': rgb_to_hex(255,255,255)})

    
    # List of columns to exclude
    #exclude_columns = ["MeanEpochsHBO", "MeanEpochsHBR", "response"]
    out_df = perf_df
    
    #.drop(columns=exclude_columns)
    
        
    
    # Move columns to the end
    """
    cols_at_end = ["MeanEqualityScore_DetectionTest","StdEqualityScore_DetectionTest", 'Relative Entropy']
    perf_df = perf_df[[c for c in perf_df if c not in cols_at_end] 
            + [c for c in cols_at_end if c in perf_df]]
    """
    col_order = ["Run", 
                 "@StartTime", 
                 "RunTime", 
                 "SubjectID",
                 "Protocol",
                 "Token",  
                 "Level", 
                 "TestType", 
                 "Region", 
                 "Side", 
                 "NoofTrials",
                'No of equality tests',
                "MI_mean", 
                "MI_median",
                "PearsonCorr_mean",
                "PearsonCorr_median",
                "SpearmanCorr_mean", 
                "SpearmanCorr_median",
                "PairwiseMI_mean",
                "PairwiseMI_std",
                "TemporalEntropy_mean", 
                "TemporalEntropy_std", 

                'Sigma_var_Init',
                "Sigma_var_firsthalf", 
                "Sigma_var_secondhalf",
                'ell_opt',
                'Sigma_f_opt',
                'Sigma_y_opt',
                "negLogLik_PSR",  
                "mean_ctrl_likelihood", 
                "var_ctrl_likelihood", 
                "optim_status",
                 "MeanEqualityScore_DetectionTest", 
                 "StdEqualityScore_DetectionTest", 
                 "StatisticalScore",
                 
                 "DetectionStatus", 
                 "Confidence",                  
                 "NewDetectionStatus", 
                 "NewConfidence", 
                 
                 "NormalisedStatisticalScore", 
                 "NonnormalisedDetectionStatus",
                 "NonnormalisedConfidence",
                 
                 "Stop", 
                 "Stop_new",
                 "Stop_nonnormalised" ]
    
                           
                           
                           
                           
                           
    col_order = [col for col in col_order if col in out_df.columns]
    out_df = out_df[col_order]

    
    # Write column headers to the Excel file
    for i, col in enumerate(out_df.columns):
        worksheet.write(0, i, col, header_format)
    
    worksheet.write(0, i+1, "Legend", header_format)
    worksheet.write(1, i+1, "Control", control_colour)
    worksheet.write(2, i+1, "PSR", psr_colour)
    
    # Determine min and max y 
    min_y = np.inf
    max_y = -np.inf
    for i in range(len(out_df)):
        curr_row_full = perf_df.iloc[i]
        if 'response' in perf_df.columns and isinstance(curr_row_full['response'], (np.ndarray)):
            curr_min = np.min(curr_row_full['response'])
            curr_max = np.max(curr_row_full['response'])
            min_y = min(min_y, curr_min)
            max_y = max(max_y, curr_max)
    # Add padding
    padding = (max_y - min_y) * 0.02
    min_y = min_y - padding
    max_y = max_y + padding

    data_row_index = 1
    i = 0
    while i < len(out_df):
        curr_row = out_df.iloc[i]
        curr_row_full = perf_df.iloc[i]
        if i + 1 < len(out_df) and ('@StartTime' in out_df.columns) and (out_df.iloc[i]['@StartTime'] == out_df.iloc[i+1]['@StartTime']) and (out_df.iloc[i]['Side'] == out_df.iloc[i+1]['Side']) and (out_df.iloc[i]['Token'] == out_df.iloc[i+1]['Token']):
            next_row = out_df.iloc[i+1]
            next_row_full = perf_df.iloc[i+1]
            for j in range(len(out_df.columns)):
                column_type = out_df.columns[j]
                f1 = None
                f2 = None
                if (pd.isna(curr_row['TestType']) or pd.isna(next_row['TestType'])):
                    if (column_type == 'Run'):
                        f1 = red_format
                        f2 = red_format
                else:
                    if (column_type == 'Run'):
                        f1 = green_format
                        f2 = green_format
                    elif (column_type == 'TestType' and curr_row[j] is not None and next_row[j] is not None):
                        f1 = yellow_format
                        f2 = blue_format
                        
                worksheet.write(data_row_index, j, get_cell_value(curr_row[j]), f1)
                worksheet.write(data_row_index + 1, j, get_cell_value(next_row[j]), f2)
    
            worksheet.set_row(data_row_index, None, border_format_top)
            worksheet.set_row(data_row_index + 1, None, border_format_bot)
            data_row_index += 2
            i += 2
        else:
            for j in range(len(out_df.columns)):
                worksheet.write(data_row_index, j, get_cell_value(curr_row[j]), red_format if j==0 else None)
            data_row_index += 1
            i += 1
    
        if (plot==True and 'response' in perf_df.columns and isinstance(curr_row_full['response'], (np.ndarray)) and isinstance(next_row_full['response'], (np.ndarray))):        
            # Write meta text
            for j in range(len(out_df.columns)):
                worksheet.write(data_row_index, j, get_cell_value(next_row[j]), white_text)
            
            # Calculate target figure size in inches
            dpi = 90  # dots per inch
            target_height_pixels = 290
            target_height_inch = target_height_pixels / dpi  # convert from pixels to inches
            
            # Assume a given aspect ratio (e.g., 16:9)
            aspect_ratio = 16 / 9
            target_width_inch = target_height_inch * aspect_ratio
            
            # Create a plot with a specific figure size
            fig, ax = plt.subplots(figsize=(target_width_inch, target_height_inch), dpi=dpi)
            
            t = np.arange(len(curr_row_full['response']))
            
            if (raw is not None):
                sfreq = raw['sfreq']
                t = t / sfreq
                
            if (epoch_bounds is not None):
                t = t + epoch_bounds[0]
                
            ax.plot(t, curr_row_full['response'], label='Control Mean', linewidth=2, color=c_hex)
            ax.plot(t, next_row_full['response'], label='PSR Mean', linewidth=2, color=psr_hex)
    # Add HBR here; MeanEpochsHBR; MeanEpochsHBO
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Intensity')
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(min_y, max_y)
            plt.suptitle(f"{next_row_full['Token']} | Trial {next_row_full['NoofTrials']} | {next_row_full['Side']} {next_row_full['Region']} | oldConf: {round(next_row_full['Confidence'], 3)} | newConf: {round(next_row_full['NewConfidence'], 3)} | Stop: {next_row_full['Stop_nonnormalised']}", fontsize=10)
            ax.axhline(0, color='grey', linewidth=2, linestyle='dotted')
            ax.axvline(0, color='grey', linewidth=2, linestyle='dotted')
            
            plt_plot = os.path.join(plot_dir, f"plot_{i}.png")
            
            
            # Then you can save your plot
            plt.savefig(plt_plot, dpi=dpi)  # Here, again you can adjust DPI if needed

            worksheet.set_row(data_row_index, 250)
    
            # Merge cells for the plot
            #worksheet.merge_range(data_row_index, 0, data_row_index, len(out_df.columns) - 1, "")
            worksheet.insert_image(data_row_index, 0, plt_plot, {'x_offset': 0, 'y_offset': 10})
            data_row_index += 1
    
    workbook.close()
    
    if (autoOpen):
        p = Popen(save_file_path, shell=True)

def runLiveSim(raw_all, save_failed_runs = True): 
    
    # Check raw validity 
    if (len(raw_all['trigger_samples']) == 0):
        raise ValueError("No events/stimuli in raw file, quitting execution...")
    
    #raw = mask_side(raw_all, side_to_use=side_to_use)
    
    fb_prms = single_run_profile(raw_all,
                                 save_failed_runs=save_failed_runs)
    while fb_prms['finished'] == False:
        
        fb_prms = single_run_profile(raw_all,
                                     save_failed_runs=save_failed_runs,
                                     fb_prms = fb_prms)

    # Concat outputs
    perf_df = pd.concat(fb_prms["perf_df_list"])

    
    return perf_df

    
if (__name__ == '__main__'):
    
    from tkinter import filedialog
    from tkinter import Tk
    
    #%% Set parameters
    link = "https://bionicsinstitute-my.sharepoint.com/:u:/g/personal/sanjayana_bionicsinstitute_org/EXlHS-odwOdLj7hZkSAN5wMB3YNCuHNBomhWw6PabMdaww"
    download_link = link + "?download=1"
    protocols = load_online_json(download_link)
    
    protocol = protocols['S012_S013']

    #nirx_file_str = r"O:\HUM\Projects\fNIRS\Arj\TestingData\Detection_2\30_C111.S012_S013.F.nosilence.yeshabituation\S012_S013.2023-02-08_005"
    nirx_file_str = r'O:\HUM\Projects\fNIRS\Arj\TestingData\HI_misc\C105_2.S012.F.AN.unmasked.modified.yessilence\S012.2023-03-17_004_unmasked\merged.pkl'
    side_to_use = "L"; # "L" or "R"

    save_failed_runs = True
    
    plotting = True
    
    # Read raw
    raw = read_nirx(nirx_file_str, protocol)
    

    #%% Run live sim
    perf_df = runLiveSim(raw, side_to_use, save_failed_runs)
    
    #%% Save file
    save_file_path = filedialog.asksaveasfilename(filetypes = [("Excel file(*.xlsx)","*.xlsx")], defaultextension = [("Excel file(*.xlsx)","*.xlsx")])
    #%%
    save_profile(perf_df, save_file_path, plot=True)
    