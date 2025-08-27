import os
from threading import Thread, Event
from queue import Full, Empty
from torch.utils.data import DataLoader
import multiprocessing as mp
import time

pause_event = Event()

class AsynchronousLoader(object):
    def __init__(self, dataset, shards, device, sampler, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, queue_size=500, drop_last=True):
        self.dataset = dataset
        self.device = device
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.queue_size = queue_size
        self.sampler = sampler
        self.shards = shards

        # Create a dataloader for the dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler, 
            shuffle=shuffle, 
            pin_memory=pin_memory, 
            num_workers=num_workers, 
            drop_last=drop_last
        )

        # Bounded queue to hold batches, with a maximum size to control memory usage
        self.queue = mp.Queue(maxsize=queue_size)
        self.idx = 0
        self.worker = None  # To track the worker thread

    def load_loop(self):
        """ Continuously load data into the queue. """
        print("CPU Loading Thread started.")
        while True:
            for i, sample in enumerate(self.dataloader):
                print(f"CPU: Putting batch {i} into the queue {id(self.queue)} (Queue size before put: {self.queue.qsize()}) ")
                self.queue.put(sample, timeout=5)  # Put data into the queue
                print(f"CPU:  batch {i} added to the queue {id(self.queue)} (Queue size after put: {self.queue.qsize()})")
            pause_event.set()  # Set the event to signal completion
            break  
            print("All batches have been loaded into the queue.")
           

    def __iter__(self):
        """ Start the worker thread and reset the iterator index. """
        self.idx = 0
        if self.worker is None or not self.worker.is_alive():
            print("Starting the CPU loading thread...")
            self.worker = Thread(target=self.load_loop)
            self.worker.daemon = True  # Ensure the thread ends with the main program
            self.worker.start()  # Start loading thread in the background
        return self

    def __next__(self):
        """ Fetch the next batch from the queue for the GPU to process. """
        while True:
            if (not self.worker.is_alive() and self.queue.empty()) or self.idx >= len(self.dataloader):
                print("No more batches to process (Queue size: ", self.queue.qsize(), ")")
                raise StopIteration  # Stop iteration when all batches are processed
            else:
                try:
                    print(f"GPU: Attempting to pull from queue {id(self.queue)}  (Queue size: { self.queue.qsize() } )")
                    batch = self.queue.get(timeout=5)  # Timeout to avoid hanging
                    print(f"GPU: Pulled batch  from queue {id(self.queue)}  (Queue size: { self.queue.qsize() } )")
                    self.idx += 4  # Increment the batch index
                    return batch
                except Empty:
                    print("GPU: Queue is empty, retrying after a brief sleep...")
                    time.sleep(0.1)  # Sleep briefly and retry if the queue is empty

    def __len__(self):
        """ Return the length of the DataLoader. """
        return len(self.dataloader)
