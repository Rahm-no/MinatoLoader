import os
import torch

from math import ceil
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
import csv
from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

from data_loading.nvidia_daliloader import get_data_loaders

from runtime.training import train
from runtime.inference import evaluate
from runtime.arguments import PARSER
from runtime.distributed_utils import init_distributed, get_world_size, get_device, is_main_process, get_rank
from runtime.distributed_utils import seed_everything, setup_seeds
from runtime.logging import get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log, mlperf_run_param_log
from runtime.callbacks import get_callbacks

DATASET_SIZE = 168



def main():
    throughput_file = "prefetch_queue_depth8.csv"
    with open(throughput_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'iteration','throughput(MBs)', 'iteration_time', 'time_diff', 'iter_persec'])
    

    torch.cuda.empty_cache()

    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet3d.log'))
    mllog.config(filename=os.path.join("/results", 'unet3d.log'))
    
    




    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    mllog_start(key=constants.INIT_START)

    flags = PARSER.parse_args()

    dllogger = get_dllogger(flags)
    local_rank = flags.local_rank
    device = get_device(local_rank)
    is_distributed = init_distributed()
    world_size = get_world_size()
    local_rank = get_rank()
    mllog_event(key='world_size', value=world_size, sync=False)
    mllog_event(key='local_rank', value=local_rank, sync=False)

    worker_seeds, shuffling_seeds = setup_seeds(flags.seed, flags.epochs, device)
    worker_seed = worker_seeds[local_rank]
    seed_everything(worker_seed)
    # mllog_event(key=constants.SEED, value=flags.seed if flags.seed != -1 else worker_seed, sync=False)

    if is_main_process:
        mlperf_submission_log()
        mlperf_run_param_log(flags)

    callbacks = get_callbacks(flags, dllogger, local_rank, world_size)
    flags.seed = worker_seed
    model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)

    mllog_end(key=constants.INIT_STOP, sync=True)
    mllog_start(key=constants.RUN_START, sync=True)

    train_dataloader, val_dataloader = get_data_loaders(flags, num_shards=world_size, global_rank=local_rank)
    # mllog_event(key='len train_dataloader', value=len(train_dataloader), sync=False)
    # mllog_event(key='len val_dataloader', value=len(val_dataloader), sync=False)

    # samples_per_epoch = world_size * len(train_dataloader) * flags.batch_size
    # mllog_event(key='samples_per_epoch', value=samples_per_epoch, sync=False)
    # mllog_event(key='batch size', value= flags.batch_size, sync=False)

    # flags.evaluate_every = flags.evaluate_every or ceil(20*DATASET_SIZE/samples_per_epoch)
    # flags.start_eval_at = flags.start_eval_at or ceil(1000*DATASET_SIZE/samples_per_epoch)
    flags.evaluate_every = 50
    flags.start_eval_at = 50

    print("flags.batch_size", flags.batch_size)
    print("world_size", world_size)
    print("flags.ga_steps", flags.ga_steps)

    mllog_event(key=constants.GLOBAL_BATCH_SIZE, value=flags.batch_size * world_size * flags.ga_steps, sync=False)
    mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=flags.ga_steps)
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout,
                         include_background=flags.include_background)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout=flags.layout,
                         include_background=flags.include_background)

    if flags.exec_mode == 'train':
        train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn, 
              device=device, callbacks=callbacks, is_distributed=is_distributed, throughput_file=throughput_file)

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_dataloader, loss_fn, score_fn,
                                device=device, is_distributed=is_distributed)
        if local_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])
    else:
        print("Invalid exec_mode.")
        pass

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
