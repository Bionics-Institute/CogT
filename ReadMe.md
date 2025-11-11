===========Folder setup instructions to import preprocessing files as module from runners test_all.py ====


1) Point Python at CoGT for this session

In the same terminal you’ll run Python from:

$env:PYTHONPATH="C:\path\to\CoGT"


Now verify:

python -c "import os,sys; print('PYTHONPATH=', os.environ.get('PYTHONPATH')); print([p for p in sys.path if 'CoGT' in p])"


You should see your CoGT path there.

2) Make sure these files exist

Create empty package markers (very important):

CoGT\fnirs_preproc\__init__.py
CoGT\fnirs_preproc\src\__init__.py


(Without these, fnirs_preproc won’t be importable as a package.)

3) Re-test importability

Run:

python -c "import importlib.util; print('pkg=', importlib.util.find_spec('fnirs_preproc')); print('sub=', importlib.util.find_spec('fnirs_preproc.src')); print('mod=', importlib.util.find_spec('fnirs_preproc.src.nirs_read_raw'))"


All three should be non-None.

4) Run your code

Keep your original import in the runner (inside CoGT\runners\src\...py):

from fnirs_preproc.src.nirs_read_raw import read_nirx

5) Make it permanent (choose ONE)

A. User env var (applies to new terminals/VS sessions):

setx PYTHONPATH "C:\path\to\CoGT"


Then close & reopen Visual Studio (and its terminal) before running.

B. Visual Studio project-only (clean):


==================================================Generate Correlation Values from fNIRs==================================================================================
0.  Set input folders in the following folder: base_folder -> SubjectNo -> subfolder (eyes closed) -> NIRx files 
1.  Change input (base_folder location) & output folders in test_all.py
2.  Run test_all.py (output: correlation values for dlpfc region measured for each subject in input folder.)
3.  Run concatenateFiles.py