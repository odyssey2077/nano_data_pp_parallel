# llmsys_s24_hw4
Public repository for Assignment 4 of 11-868 LLM Systems.

python project/run_data_parallel.py --world_size 1 --batch_size 64 --n_epochs 5
Rank 0 training time: avg:34.295407152175905, std:0.16795867448069426,         tokens_per_second: avg: 153666.3878232988, std:1827.136747259087

python project/run_data_parallel.py --world_size 2 --batch_size 128 --n_epochs 5
Rank 0 training time: avg:19.617389392852782, std:0.10288876462163467,         tokens_per_second: avg: 151388.9372866468, std:2724.7173001005117
Rank 1 training time: avg:18.995491075515748, std:0.29851170460411725,         tokens_per_second: avg: 150326.2330268352, std:2444.400030953019