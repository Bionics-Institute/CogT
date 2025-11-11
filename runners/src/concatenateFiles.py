import pandas as pd
import glob
import os

# Folder containing your Excel files
folder_path = r"C:\\Users\\GBalasubramanian\\OneDrive - The Bionics Institute of Australia\\Documents\\FMRI_FNIRS_Work\\Experiments\\fnirsCorrelationValues\\output\\individual subjects"

# Get all Excel files in the folder
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

# Read and concatenate them automatically
df_list = [pd.read_excel(f) for f in excel_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Optionally save combined file
combined_df.to_excel(os.path.join(folder_path, "combined_output.xlsx"), index=False)

print(f"Combined {len(excel_files)} files successfully.")
