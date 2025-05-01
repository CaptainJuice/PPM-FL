filename=exp_protein_run_rips

### Data settings ###
# Name of the dataset. The python code will access to "data/{dataset}_data" and "data/{dataset}_label".
dataset="KNproteinNoisy01_C=7,T=500,K=60"

### PH settings: You can use zero or one of the following three options. ###
rips=0 # If 1, DistMatrixNet will be used instead of PH. If 0, this will not be used. 
toporep=1 # If 1, proposed method will be used. If 0, this will not be used. 
dtm=0 # If >= 1, DTM filtration with k=dtm will be used. Especially, if dtm=1, Rips filtration will be used. If 0, this will not be used. 
nb_repeat=200
dim=0
bs=128

### Name of the directory to save ###
savedirname="result/${dataset}_rips${rips}_tr${toporep}_dtm${dtm}_PPM_repeat${nb_repeat}_dim${dim}_bs${bs}_rips"

mkdir $savedirname
python3 $filename.py $savedirname $dataset $rips $toporep $dtm $nb_repeat $dim $bs