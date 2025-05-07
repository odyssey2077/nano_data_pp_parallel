from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
import math
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module
import time
# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    schedule = [(0, 0)]
    while True:
        yield schedule
        if len(schedule) == 1 and schedule[0] == (num_batches - 1, num_partitions - 1):
            break
        else:
            next_schedule = []
            for i, j in schedule:
                if j == 0 and i < num_batches - 1:
                    next_schedule.append((i+1, 0))
                if j < num_partitions - 1:
                    next_schedule.append((i, j+1))
            schedule = next_schedule
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN SOLUTION
        batch_size = x.size(0)
        num_microbatches = math.ceil(batch_size / self.split_size)
        batches = list(x.chunk(num_microbatches, dim=0))
        self.schedule = list(_clock_cycles(len(batches), len(self.partitions)))
        for schedule in self.schedule:
            self.compute(batches, schedule)
        return torch.cat(batches, dim=0)
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN SOLUTION
        for mb_index, partition in schedule:
            cur_batch = batches[mb_index]
            cur_partition = partitions[partition]
            batches[mb_index] = batches[mb_index].to(devices[partition])
            task = Task(lambda: cur_partition(cur_batch))
            self.in_queues[partition].put(task)


        for mb_index, partition in schedule:
            succeed, result = self.out_queues[partition].get()
            if not succeed:
                print(f"Error in partition {partition}")
                print(result)
            else:
                batches[mb_index] = result[1]
        # END SOLUTION

