import pytest
from pipeline.pipe import _clock_cycles, Pipe
from pipeline.partition import _split_module, WithDevice
from torch import nn
import torch

@pytest.mark.a4_2_1
def test_clock_cycles_0():
    m = 6
    n = 2
    gold_schedule = iter([[(0, 0)], 
        [(1, 0), (0, 1)], 
        [(2, 0), (1, 1)],
        [(3, 0), (2, 1)],
        [(4, 0), (3, 1)],
        [(5, 0), (4, 1)],
        [(5, 1)]]
    )
    for schedule in _clock_cycles(m, n):
        assert sorted(schedule) == sorted(next(gold_schedule))

@pytest.mark.a4_2_1
def test_clock_cycles_1():
    m = 3
    n = 3
    gold_schedule = iter([[(0, 0)], 
        [(1, 0), (0, 1)], 
        [(2, 0), (1, 1), (0, 2)],
        [(2, 1), (1, 2)],
        [(2, 2)]]
    )
    for schedule in _clock_cycles(m, n):
        assert sorted(schedule) == sorted(next(gold_schedule))


# Test cases using pytest parameterization
@pytest.mark.a4_2_1
@pytest.mark.parametrize(
    "num_batches, num_partitions, expected_schedules",
    [
        (1, 1, [
            [(0, 0)]
        ]),
        (2, 2, [
            [(0, 0)],
            [(1, 0), (0, 1)],
            [(1, 1)]
        ]),
        (3, 3, [
            [(0, 0)],
            [(1, 0), (0, 1)],
            [(2, 0), (1, 1), (0, 2)],
            [(2, 1), (1, 2)],
            [(2, 2)]
        ]),
        (1, 3, [
            [(0, 0)],
            [(0, 1)],
            [(0, 2)]
        ]),
        (3, 1, [
            [(0, 0)],
            [(1, 0)],
            [(2, 0)]
        ]),
        (2, 3, [
            [(0, 0)],
            [(1, 0), (0, 1)],
            [(1, 1), (0, 2)],
            [(1, 2)]
        ]),
        (3, 2, [
            [(0, 0)],
            [(1, 0), (0, 1)],
            [(2, 0), (1, 1)],
            [(2, 1)]
        ]),
    ]
)
def test_clock_cycles(num_batches, num_partitions, expected_schedules):
    """
    Tests the _clock_cycles function against known correct outputs for various inputs.
    """
    generated_schedules = list(_clock_cycles(num_batches, num_partitions))

    # It's important to sort the inner lists (schedules) before comparison
    # because the order of tuples within a schedule might not be guaranteed
    # by the function's logic, but the set of tasks in a clock cycle should match.
    sorted_generated = [sorted(schedule) for schedule in generated_schedules]
    sorted_expected = [sorted(schedule) for schedule in expected_schedules]

    assert sorted_generated == sorted_expected, \
        f"For batches={num_batches}, partitions={num_partitions}:\n" \
        f"Expected: {sorted_expected}\n" \
        f"Got: {sorted_generated}"


@pytest.mark.a4_2_1
def test_split_module_0():
    model = nn.Sequential(
          nn.Conv2d(10,20,5).to('cuda:0'),
          nn.Conv2d(20,64,5).to('cuda:0'),
          nn.Conv2d(64,128,5).to('cuda:1'),
    )
    partitions, devices = _split_module(model)
    assert len(partitions) == 2
    assert len(devices) == 2
    assert len(partitions[0]) == 2
    assert next(partitions[0].parameters()).device == devices[0]
    assert next(partitions[1].parameters()).device == devices[1]

@pytest.mark.a4_2_1
def test_split_module_1():
    model = nn.Sequential(
          nn.Conv2d(10,20,5).to('cuda:0'),
          WithDevice(nn.Dropout(0.5), 'cuda:0'),
          nn.Conv2d(20,64,5).to('cuda:1'),
    )
    partitions, devices = _split_module(model)
    assert len(partitions) == 2
    assert len(devices) == 2
    assert next(partitions[0].parameters()).device == devices[0]
    assert next(partitions[1].parameters()).device == devices[1]

@pytest.mark.a4_2_2
# @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
# @pytest.mark.parametrize("split_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("split_size", [8])
def test_forward_0(batch_size, split_size):
    # update first layer to be all 2, second layer to be all 3
    model = nn.Sequential(
        nn.Linear(3, 4).to('cuda:0'),
        WithDevice(nn.Sigmoid(), 'cuda:0'),
        nn.Linear(4, 5).to('cuda:0'),
        WithDevice(nn.Sigmoid(), 'cuda:0'),
    )
    model[0].weight.data = torch.ones(3, 4) * 2
    model[2].weight.data = torch.ones(4, 5) * 3
    model[0].bias.data = torch.zeros(4)
    model[2].bias.data = torch.zeros(5)
    model[0].to('cuda:0')
    model[2].to('cuda:0')
    # update x to be all 1
    x = torch.ones(batch_size, 3).to('cuda:0')
    y0 = model(x).to('cpu')

    # move the last two layer to another device
    model[-2] = model[-2].to('cuda:1')
    model[-1] = WithDevice(nn.Sigmoid(), 'cuda:1')
    pipe = Pipe(model, split_size=split_size)
    y1 = pipe(x).to('cpu')
    assert torch.allclose(y0, y1)
