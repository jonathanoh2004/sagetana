# Only areas marked with *CHANGE* are necessary for change for a bare minimum run!

# print formatting
print_message() {
  echo "########################"
  echo $1
  echo "########################"
}

# *CHANGE* DATA_P OUTPUT_P
# base paths defined here to avoid clutter (change these to match personal paths)
# DATA_P is path to folder where your data is stored
# OUTPUT_P is path to folder where your output will be stored 
DATA_P="/Users/sudarshan/Documents/Jonathan/data/data1"
OUTPUT_P="/Users/sudarshan/Documents/Jonathan/outputs/reg"

# *CHANGE* Change the filename and TEs!
# Define the echos and TEs by adding the file name and times
# If you set up the DATA_P correctly shouldn't need anything but the file name!
ECHO1="$DATA_P/Multigre_SAGE_e1_tshift_bet.nii.gz"
ECHO2="$DATA_P/Multigre_SAGE_e2_tshift_bet.nii.gz"
ECHO3="$DATA_P/Multigre_SAGE_e3_tshift_bet.nii.gz"
ECHO4="$DATA_P/Multigre_SAGE_e4_tshift_bet.nii.gz"
ECHO5="$DATA_P/Multigre_SAGE_e5_tshift_bet.nii.gz"
TE="7.3 26.4 57.5 76.6 95.7"

# Script assumes that you select e1 for your masks
# The Code to create these files is right below but we are "predicting" the names here
# Since we are not gathering existing files, but instead just setting names for the new files we dont need to change anything here from run to run.
TMEAN="$DATA_P/e1_tmean.nii.gz"
BRAIN="$DATA_P/e1_tmean_brain.nii.gz"
MASK="$DATA_P/e1_tmean_brain_mask.nii.gz"

# runs fslmaths to get 3d image
# lines 37 and 38 should create 3 new files inside the folder you chose with DATA_P
print_message "Creating mask for echo 1"
fslmaths $ECHO1 -Tmean $TMEAN
bet $TMEAN $BRAIN -m
print_message "Mask was Created Successfully"

# print the parameters for running sage-tedana
print_message "Running tedana:"
echo "Echo 1: $ECHO1"
echo "Echo 2: $ECHO2"
echo "Echo 3: $ECHO3"
echo "Echo 4: $ECHO4"
echo "Echo 5: $ECHO5"
echo "TE: $TE"
echo "Output Directory: $OUTPUT_P"
echo "Mask File: $MASK"

# Here is where we actually execute tedana
# Line 55 is the one that is actually taking all the variables and running the sage-tedana!
print_message "Executing tedana"
tedana -d $ECHO1 $ECHO2 $ECHO3 $ECHO4 $ECHO5 -e $TE --out-dir $OUTPUT_P --mask $MASK --overwrite
print_message "done."
