# PMN (Paired real data Meet Noise model)
Due to capacity constraints, we cannot upload all the data directly. We are rewriting data preprocessing and data I/O.
You can look at the code section first, SNA's parameter sampling is implemented as the `raw_wb_aug_torch` function in the file ```data_process/process.py```.  
The complete process of SNA has the CPU version in the `Mix_Dataset` class in ```data_process/real_datasets.py``` and the GPU version in the `preprocess` function in ```trainer_SID.py```.

## TODO list
* [√] code release
* [√] code can run for evaluation
* [√] user guide for evaluation
* [ ] code can run for training
* [ ] user guide for training

## Prerequisites
* Python >=3.6, PyTorch >= 1.6
* Requirements: opencv-python, rawpy, exifread, h5py, scipy
* Platforms: Ubuntu 16.04, cuda-10.1 (It)

Please download the ELD dataset and SID dataset first, which are necessary for validation (or training).   
ELD ([official project](https://github.com/Vandermode/ELD)): [ELD (11.46 GB)](https://drive.google.com/file/d/13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX/view?usp=sharing)  
SID ([official project](https://github.com/cchen156/Learning-to-See-in-the-Dark)):  [Sony (25 GB)](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)

## Quick Start
1. use `get_dataset_infos.py` to generate dataset infos
```bash 
# Evaluate
python3 get_dataset_infos.py --dstname ELD --root_dir /data/ELD --mode SonyA7S2
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode evaltest
# Train
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode train
```
2. evaluate
```bash 
# You can replace 'Ours.yml' to other configuration to change methods.
# If you don't want to save pictures, please use add '--save_plot False'. This option will save your time and space.
# ELD & SID
python3 trainer_SID.py -f runfiles/Ours.yml --mode evaltest
# ELD only
python3 trainer_SID.py -f runfiles/Ours.yml --mode eval
# SID only
python3 trainer_SID.py -f runfiles/Ours.yml --mode test
```
3. train
```bash 
# you can replace 'Ours.yml' to other configuration to change methods.
python3 trainer_SID.py -f runfiles/Ours.yml --mode train
```