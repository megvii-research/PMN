# PMN (Paired real data Meet Noise model)
Due to capacity constraints, we cannot upload all the data directly. We are rewriting data preprocessing and data I/O.
You can look at the code section first, SNA's parameter sampling is implemented as the raw_wb_aug_torch function in the file 'data_process/process.py'.

The complete process of SNA has the CPU version in the Mix_Dataset class in 'data_process/real_datasets.py' and the GPU version in the preprocess function in 'trainer_SID.py'.

## TODO list
* [√] code release
* [√] code can run for evaluation
* [ ] user guide for evaluation
* [ ] code can run for training
* [ ] user guide for training