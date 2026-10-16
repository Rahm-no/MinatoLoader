"""CPU-only checks that the asynchronous loader draws a new sample order every epoch.

Run from the repo root: python -m pytest Minato/tests
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loading import Asynchrnous_dataloader as async_loader  # noqa: E402


class Toy:
    """Sample i is filled with the value i, so a batch shows which samples it holds."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        sample = np.full((1, 2, 2, 2), float(i), dtype=np.float32)
        return sample, sample.copy()


def producers(dataset, sampler, shuffle, num_workers=2):
    return [async_loader.DataProducer(None, dataset, 2, shuffle, False, "cpu", 50, [], sampler, num_workers, i, 0)
            for i in range(num_workers)]


def test_producers_split_a_new_distributed_order_every_pass():
    dataset = Toy(32)
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=0, drop_last=True)
    orders = []
    for epoch in range(1, 4):
        slices = [producer.epoch_indices(epoch) for producer in producers(dataset, sampler, False)]
        sampler.set_epoch(epoch)
        assert slices[0] + slices[1] == list(sampler)
        orders.append(tuple(slices[0] + slices[1]))
    assert len(set(orders)) == 3


def test_producers_split_a_new_random_order_every_pass():
    dataset = Toy(32)
    orders = []
    for epoch in range(1, 4):
        slices = [producer.epoch_indices(epoch) for producer in producers(dataset, RandomSampler(dataset), True)]
        assert sorted(slices[0] + slices[1]) == list(range(32))
        orders.append(tuple(slices[0] + slices[1]))
    assert len(set(orders)) == 3


def test_order_without_shuffle_is_unchanged():
    dataset = Toy(32)
    producer = async_loader.DataProducer(None, dataset, 2, False, False, "cpu", 50, list(range(16)),
                                         SequentialSampler(dataset), 2, 0, 0)
    assert producer.epoch_indices(1) == list(range(16))


@pytest.mark.skipif(sys.platform == "darwin", reason="multiprocessing.Queue.qsize() is not implemented on macOS")
def test_loader_order_changes_between_epochs():
    os.environ["WORLD_SIZE"] = "2"
    mp.set_start_method("fork", force=True)
    torch.manual_seed(0)
    dataset = Toy(32)
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=0, drop_last=True)
    loader = async_loader.AsynchronousLoader(dataset, device="cpu", shards=2, rank=0,
                                             slow_processed_queue=mp.Queue(), batch_size=2,
                                             shuffle=False, pin_memory=False, num_workers=1, sampler=sampler)
    orders = []
    try:
        for epoch in range(1, 5):
            loader.sampler.set_epoch(epoch)  # as runtime/training.py does
            orders.append(tuple(int(v) for images, _ in loader for v in images[:, 0, 0, 0, 0].tolist()))
    finally:
        async_loader.data_stop_event.set()
        for producer in loader.producers:
            producer.terminate()
            producer.join(timeout=5)
    assert all(sorted(order) == sorted(set(order)) and len(order) == 16 for order in orders)
    assert len(set(orders)) == 4
