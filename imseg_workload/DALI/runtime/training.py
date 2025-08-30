
from tqdm import tqdm
import os
import time
import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler
import csv
from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS


def get_optimizer(params, flags):
    if flags.optimizer == "adam":
        optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
    elif flags.optimizer == "sgd":
        optim = SGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
                    weight_decay=flags.weight_decay)
    elif flags.optimizer == "lamb":
        import apex
        optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas, 
                                          weight_decay=flags.weight_decay)
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim


def calculate_tensor_size(tensor):
    print("tensor.shape", tensor.shape)
    num_elements = tensor.numel()  # Total number of elements
    bytes_per_element = tensor.element_size()  # Size of each element in bytes
    size_in_bytes = num_elements * bytes_per_element
    size_in_mb = size_in_bytes / (1024 ** 2)  # Convert bytes to MB
    print(f"Tensor size: {size_in_mb:.2f} MB")
    return size_in_mb


def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr + (lr - init_lr) * scale


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed, throughput_file):
    rank = get_rank()
    start_training_time = time.time()
    torch.cuda.empty_cache()

    world_size = get_world_size()
    torch.backends.cudnn.benchmark = flags.cudnn_benchmark
    torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    optimizer = get_optimizer(model.parameters(), flags)
    if flags.lr_decay_epochs:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=flags.lr_decay_epochs,
                                                         gamma=flags.lr_decay_factor)
    scaler = GradScaler()

    model.to(device)
    loss_fn.to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank],
                                                          output_device=rank)

    is_successful = False
    diverged = False
    next_eval_at = flags.start_eval_at
    model.train()

    for epoch in range(1, flags.epochs + 1):
          
        cumulative_loss = []
        n=0
        size = 0 
        last_n=0
        last_time = time.time()

        if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
            lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
        mllog_start(key=CONSTANTS.BLOCK_START, sync=False,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1})
        mllog_start(key=CONSTANTS.EPOCH_START, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

        # if is_distributed:
        #     train_loader.sampler.set_epoch(epoch)

        loss_value = None
        optimizer.zero_grad()
        torch.cuda.synchronize()  # Ensure all CUDA ops are done
        start_prep = time.time()
        total_prep_time = 0
        total_train_time = 0

        for iteration, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
            print("iteration", iteration, "epoch", epoch, "dataloader length", len(train_loader), "length batch", len(batch))
            image = torch.cat([d['data'] for d in batch], dim=0).to(device)
            label = torch.cat([d['label'] for d in batch], dim=0).to(device)
        
         
            d = batch[0]
            size = size + calculate_tensor_size(d['data']) + calculate_tensor_size(d['label'])

            print(f"Image dtype: {image.dtype}, shape: {image.shape}")
            print(f"Label dtype: {label.dtype}, shape: {label.shape}")
       
            torch.cuda.synchronize()
            prep_time = time.time() - start_prep
            total_prep_time += prep_time
            start_train = time.time()
           
            

      

            with autocast(enabled=flags.amp):
                output = model(image)
               

                loss_value = loss_fn(output, label)
                loss_value /= flags.ga_steps
                print("loss_value in speedy", loss_value)

            if flags.amp:
                scaler.scale(loss_value).backward(retain_graph=True)
            else:
                loss_value.backward(retain_graph=True)

            if (iteration + 1) % flags.ga_steps == 0:
                if flags.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
            
       
                        

            loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy()
            cumulative_loss.append(loss_value)
            if time.time() - last_time > 2:
                # Calculate number of iterations per second over the last 5 seconds
                iter_persec = (n - last_n) / (time.time() - last_time + 1e-1)
                throughput = (size ) / (time.time() - last_time + 1e-1)
                print(
                    f"Epoch {epoch} Iteration {iteration} - "
                    f"Iterations per second: {iter_persec:.2f} "
                    f"n: {n} "
                  
                )
                with open(throughput_file, 'a', newline='') as f:
                    f.write(f"{epoch},{n},{throughput},{time.time() - start_training_time},{time.time() - last_time}, {iter_persec},{size}\n")
                

                last_time = time.time()
                last_n = n
                size = 0
            n = n + 1
            train_time = time.time() - start_train
            total_train_time += train_time

            # Reset prep timer for next batch
            torch.cuda.empty_cache()

            start_prep = time.time()
            with open("dali_prep_time_imseg.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch , n, prep_time, total_prep_time, train_time, total_train_time])

                


        mllog_end(key=CONSTANTS.EPOCH_STOP, sync=False,
                    metadata={CONSTANTS.EPOCH_NUM: epoch, 'current_lr': optimizer.param_groups[0]['lr']})

        if flags.lr_decay_epochs:
            scheduler.step()

        if epoch == next_eval_at:
            next_eval_at += flags.evaluate_every
            del output
            mllog_start(key=CONSTANTS.EVAL_START, value=epoch, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, epoch)
            eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
            with open(accuracy_file, 'a', newline='') as f:
                f.write(f"{epoch},{eval_metrics['mean_dice']},{eval_metrics['L1 dice']},{eval_metrics['L2 dice']}\n")
            
            mllog_event(key=CONSTANTS.EVAL_ACCURACY, 
                        value=eval_metrics["mean_dice"], 
                        metadata={CONSTANTS.EPOCH_NUM: epoch}, 
                        sync=False)
            mllog_end(key=CONSTANTS.EVAL_STOP, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=False)

        
            model.train()
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                is_successful = True
            elif eval_metrics["mean_dice"] < 1e-6:
                print("MODEL DIVERGED. ABORTING.")
                diverged = True

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=False,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1})

        if is_successful or diverged:
            break

    mllog_end(key=CONSTANTS.RUN_STOP, sync=True,
                metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS if is_successful else CONSTANTS.ABORTED})


