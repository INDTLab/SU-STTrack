# SU-STTrack


### Purely PyTorch-based Code

SU-STTrack is implemented purely based on the PyTorch. 

## Install the environment
 Use the Anaconda
```
conda create -n stark python=3.6
conda activate susttrack
bash install_pytorch17.sh
```
## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${SU-STTrack_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train SU-STTrack
Training with multiple GPUs using DDP
```
# SU-STTrack Stage1
python tracking/train.py --script susttrack_st1 --config baseline_R101 --save_dir . --mode multiple --nproc_per_node 8  
# SU-STTrack Stage2
python tracking/train.py --script susttrack_st2 --config baseline_R101 --save_dir . --mode multiple --nproc_per_node 8 --script_prv susttrack_st1 --config_prv baseline_R101 
```
## Test and evaluate SU-STTrack on benchmarks

- LaSOT
```
python tracking/test.py susttrack_st baseline --dataset lasot --threads 32
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py susttrack_st baseline_got10k_only --dataset got10k_test --threads 32
python lib/test/utils/transform_got10k.py --tracker_name susttrack_st --cfg_name baseline_got10k_only
```
- TrackingNet
```
python tracking/test.py susttrack_st baseline --dataset trackingnet --threads 32
python lib/test/utils/transform_trackingnet.py --tracker_name susttrack_st --cfg_name baseline
```
