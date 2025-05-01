filename=exp_ModelNet_run_two_phase_scale

### Data settings ###
# Name of the dataset. The python code will access to "data/{dataset}_data" and "data/{dataset}_label".
dataset="ModelNetNoisy01_C=10,N=100,T=1,K=2000"

### PH settings: You can use zero or one of the following three options. ###
rips=0 # If 1, DistMatrixNet will be used instead of PH. If 0, this will not be used. 
toporep=1 # If 1, proposed method will be used. If 0, this will not be used. 
dtm=0 # If >= 1, DTM filtration with k=dtm will be used. Especially, if dtm=1, Rips filtration will be used. If 0, this will not be used. 
nb_repeat=400
dim=1
bs=8
pointnet=0
deepsets=1
pointmlp=0


### Name of the directory to save ###
savedirname="result/${dataset}_rips${rips}_tr${toporep}_dtm${dtm}_PPM_repeat${nb_repeat}_dim${dim}_bs${bs}_pointnet${pointnet}_deepsets${deepsets}_pointmlp${pointmlp}"

mkdir $savedirname
python3 $filename.py $savedirname $dataset $rips $toporep $dtm $nb_repeat $dim $bs $pointnet $deepsets $pointmlp