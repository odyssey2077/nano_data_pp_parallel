# Parallel Training Implementation

A toy implementation for data parallel and pipeline parallel training, based on skeleton code from [llmsys_s24_hw4](https://llmsystem.github.io/llmsystem2025springhw/assignment_4/).

## Overview

This project implements and compares two parallelism strategies:

- **Data Parallel**: Distributes batches across multiple GPUs
- **Pipeline Parallel**: Splits model layers across multiple GPUs

## Implementation Notes

### Data Parallel
Data parallel is straightforward as we can directly leverage the `allreduce` operator provided by `torch.distributed`. We use multiprocessing since we don't share any memory between workers. It's worth noting that our implementation performs `allreduce` for every parameter in the model, which may not be efficient if there's no network optimization within `torch.distributed`.

### Pipeline Parallel
Implementing pipeline parallel is more challenging. One notable bug that took hours of debugging was related to passing the proper function to the Task object. Initially, using lambda functions was suggested, but this approach isn't memory safe. This experience highlighted the benefits of Rust's closure design. The solution was to use `functools.partial` to properly fix the parameters passed to functions.

**Note**: Run tests multiple times with `python -m pytest -l -v -k "a4_2_2"` as results have variance.

## Experimental Results

Experiments were run on 2 RTX 4090D GPUs rented from AutoDL.

### Data Parallel (1 GPU)

python project/run_data_parallel.py --world_size 1 --batch_size 64 --n_epochs 5

Rank 0 training time: avg:20.642004537582398, std:0.3267517407662595,         tokens_per_second: avg: 217800.29586771486, std:3162.662176843728

python project/run_data_parallel.py --world_size 2 --batch_size 128 --n_epochs 5

Rank 0 training time: avg:14.238383245468139, std:0.731379118196378,         tokens_per_second: avg: 218496.67599640283, std:10156.474085906635

Rank 1 training time: avg:13.97455563545227, std:1.646963603810814,         tokens_per_second: avg: 216126.32466847048, std:7582.167021660448

python project/run_pipeline.py --model_parallel_mode='pipeline_parallel'

Training time: avg:36.46264934539795, std:0.009329080581665039,         tokens_per_second: avg: 17552.208997004353, std:4.490786463928089

python project/run_pipeline.py --model_parallel_mode='model_parallel'

Training time: avg:21.262330532073975, std:0.3602488040924072,         tokens_per_second: avg: 30108.824369474085, std:510.13542261368275

