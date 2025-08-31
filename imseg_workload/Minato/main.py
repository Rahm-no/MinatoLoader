import os
import torch
from  imagespacing import image_spacing_ext
from math import ceil
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
import logging
from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore
import multiprocessing 
import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import concurrent.futures
import multiprocessing
import multiprocessing as mp
import csv
import pdb
import datetime
datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

import psutil
import signal

from data_loading.data_loader import get_data_split
from data_loading.pytorch_loader import PytTrain, PytVal
from runtime.training import train
import sys
import threading
import queue
from runtime.inference import evaluate
from runtime.arguments import PARSER
from runtime.distributed_utils import init_distributed, get_world_size, get_device, is_main_process, get_rank,clean
from runtime.distributed_utils import seed_everything, setup_seeds
from runtime.logging import get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log, mlperf_run_param_log
from runtime.callbacks import get_callbacks
import concurrent.futures
import torch
import torch.multiprocessing as mp 
# import torch.distributed as dist

import subprocess
import time
#def create_PytTrain_instance(x_train, y_train, **train_data_kwargs):
 #   return PytTrain(x_train, y_train, **train_data_kwargs)


import os
import torch

from math import ceil
from mlperf_logging import mllog
from mlperf_logging.mllog import constants

from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

from data_loading.data_loader import get_data_loaders

from runtime.training import train
from runtime.inference import evaluate
from runtime.arguments import PARSER
from runtime.distributed_utils import init_distributed, get_world_size, get_device, is_main_process, get_rank
from runtime.distributed_utils import seed_everything, setup_seeds
from runtime.logging import get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log, mlperf_run_param_log
from runtime.callbacks import get_callbacks

DATASET_SIZE = 168


def main():
    throughput_file = "test_minato.csv"
    with open(throughput_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch','iteration','throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec'])
    
    # accuracy_file = "accuracy_50epochsoffandon.csv"
    # with open(accuracy_file, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(['epoch', 'accuracy', 'mean dice','l1 dice','l2 dice'])


    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet3d.log'))
    mllog.config(filename=os.path.join("/results", 'unet3d.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    mllog_start(key=constants.INIT_START)
 
    
    flags = PARSER.parse_args()
    image_spacing_instance = image_spacing_ext(flags.raw_dir)
    image_spacing_dic = image_spacing_instance()  # Call the instance to extract image spacings
    flags.image_spacings = image_spacing_dic
    flags.disable_logging=True
    dllogger = get_dllogger(flags)
    local_rank = flags.local_rank
    device = get_device(local_rank)
    is_distributed = init_distributed()
    world_size = get_world_size()
    local_rank = get_rank()

    mllog_event(key='world_size', value=world_size, sync=False)
    mllog_event(key='local_rank', value=local_rank, sync=True)
    mllog_event(key='Batch size used', value=flags.batch_size, sync=True)
    mllog_event(key='Epochs', value = flags.epochs, sync=True)

    worker_seeds, shuffling_seeds = setup_seeds(flags.seed, flags.epochs, device)
    worker_seed = worker_seeds[local_rank]
    seed_everything(worker_seed)
    mllog_event(key=constants.SEED, value=flags.seed if flags.seed != -1 else worker_seed, sync=False)

    if is_main_process:
        mlperf_submission_log()
        mlperf_run_param_log(flags)

    callbacks = get_callbacks(flags, dllogger, local_rank, world_size)
    flags.seed = worker_seed
    model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)

    mllog_end(key=constants.INIT_STOP, sync=True)
    mllog_start(key=constants.RUN_START, sync=True)

    train_dataloader, val_dataloader = get_data_loaders(flags, num_shards=world_size, global_rank=local_rank, device=device, slow_queue=slow_queue, slow_processed_queue=slow_processed_queue)
    mllog_event(key='len train_dataloader', value=len(train_dataloader), sync=False)
    mllog_event(key='len val_dataloader', value=len(val_dataloader), sync=False)

    samples_per_epoch = world_size * len(train_dataloader) * flags.batch_size
    mllog_event(key='samples_per_epoch', value=samples_per_epoch, sync=False)
    flags.evaluate_every = flags.evaluate_every or ceil(20*DATASET_SIZE/samples_per_epoch)
    flags.start_eval_at = flags.start_eval_at or ceil(1000*DATASET_SIZE/samples_per_epoch)

    mllog_event(key=constants.GLOBAL_BATCH_SIZE, value=flags.batch_size * world_size * flags.ga_steps, sync=False)
    mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=flags.ga_steps)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout,
                         include_background=flags.include_background)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout=flags.layout,
                         include_background=flags.include_background)

    if flags.exec_mode == 'train':
        print("training starts here")
        t0 = time.time()
        
        train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn, 
              device=device, callbacks=callbacks, is_distributed=is_distributed, throughput_file=throughput_file)
        t1 = time.time()
        total_time = t1 - t0
     

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_dataloader, loss_fn, score_fn,
                                device=device, is_distributed=is_distributed)
        if local_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])
    else:
        print("Invalid exec_mode.")
        pass





def cleanup_processes():
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        try:
            print(f"Joining {child.pid} ({child.name()})")
            child.terminate()  # sends SIGTERM
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            print(f"Joining {child.pid}")
            child.kill()  # sends SIGKILL
        except psutil.NoSuchProcess:
            pass


if __name__ == "__main__":


    import torch.multiprocessing as mp

    slow_queue = mp.Queue()
    slow_processed_queue = mp.Queue()

    main()
    print('FINISH')
    cleanup_processes()
