# MinatoLoader

## 1. Abstract  

*Data loaders* are used by Machine Learning (ML) frameworks like **PyTorch** and **TensorFlow** to apply transformations to data before feeding it into the accelerator. This operation is called **data preprocessing**.  Data preprocessing plays an important role in the ML training workflow because if it is inefficiently pipelined with the training, it can yield **high GPU idleness**, resulting in important training delays.  Unfortunately, existing data loaders waste GPU resources — for example, the **PyTorch DataLoader** leads to about **76% GPU idleness**. A key source of inefficiency is the variability in preprocessing time across samples within the same dataset. Existing data loaders are oblivious to this variability and construct batches without considering slow vs. fast samples. As a result, the entire batch is delayed by a single slow sample, **stalling the training pipeline and causing head-of-line blocking**.   To address these inefficiencies, we present **MinatoLoader**, a general-purpose data loader for PyTorch that accelerates training and improves GPU utilization.   MinatoLoader is designed for a **single-server, multi-GPU setup**. It continuously prepares data in the background and actively constructs batches by prioritizing fast-to-preprocess samples, while slower samples are processed in parallel.  We evaluate MinatoLoader on servers with **NVIDIA V100** and **A100 GPUs**.  
- On a machine with four A100 GPUs, MinatoLoader improves training time of a wide range of workloads by up to **7.5× (3.6× on average)** over PyTorch DataLoader and Pecan, and up to **3× (2.2× on average)** over NVIDIA DALI.  
- It also increases **average GPU utilization from 46.4% (PyTorch) to 90.45%**, while preserving model accuracy and enabling faster convergence.  

## 2. Execution Environment

Our experiments were run on the following environment:

- **Operating System**: xx
- **CPU**:xx
- **RAM**: xx
- **GPU**:xx
- **Storage**: xx

**Software Stack**:
- NVIDIA Driver: xx
- CUDA: xx
- cuDNN: xx
- PyTorch: xx
- NVIDIA DALI: xx
- Python: xx
- Docker: xx

We also verified functionality on a smaller setup:
- xx
- xx

## 3. A description of each artifact component and how it relates to the paper


## 4. Introduction 
This work presents 3D-UNet workload using three systems: PyTorch, DALI and Minato. 


This benchmark represents a 3D medical image segmentation task using [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) dataset called [KiTS19](https://github.com/neheller/kits19). The task is carried out using a [U-Net3D](https://arxiv.org/pdf/1606.06650.pdf) model variant based on the [No New-Net](https://arxiv.org/pdf/1809.10483.pdf) paper.


The data is stored in the [KiTS19 github repository](https://github.com/neheller/kits19).

## Steps to download and preprocess the data
1. Clone the MinatoLoader Eurosys  repo
```bash 
git clone git@github.com:Rahm-no/MinatoLoader.git
```
2. Build docker image 
```bash 
cd imseg_workload
docker build -t minato:latest .
./start_container.sh 
```
3. Download the data
   
    To download the data please follow the instructions:
    ```bash
    mkdir raw-data-dir
    cd raw-data-dir
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    ```
    This will download the original, non-interpolated data to `raw-data-dir/kits19/data`

 
2. Preprocess the dataset.
   
    
    The data preprocessing script is called `preprocess_dataset.py`. All the required hyperparameters are already set. All you need to do is to invoke the script with correct paths:
    ```bash
    python3 preprocess_dataset.py --data_dir /raw_data --results_dir /data
    ```
   
    The script will preprocess each volume and save it as a numpy array at `/data`. It will also display some statistics like the volume shape, mean and stddev of the voxel intensity. Also, it will run a checksum on each file comparing it with the source.

<!-- 2. Start an interactive session in the container to run preprocessing/training/inference.
 
    You will need to mount two (or three) directories:
    - for raw data (RAW-DATA-DIR) 
    - for preprocessed data (PREPROCESSED-DATA-DIR)
    - (optionally) for results (RESULTS-DIR)
    
    ```bash
    mkdir data
    mkdir results
   -->
    ```
<!-- 
## Steps to run and time

The basic command to run on 1 worker takes form:
```bash
bash run_and_time.sh <SEED>
```

The script assumes that the data is available at `/data` directory.

Running this command for seeds in range `{0, 1, ..., 9}` should converge to the target accuracy `mean_dice` = 0.908. 
The training will be terminated once the quality threshold is reached or the maximum number of epochs is surpassed. 
If needed, those variables can be modified within the `run_and_time.sh` script.


## Repository content
 
In the root directory, the most important files are:
* `main.py`: Serves as the entry point to the application. Encapsulates the training routine.
* `Dockerfile`: Container with the basic set of dependencies to run U-Net3D.
* `requirements.txt`: Set of extra requirements for running U-Net3D.
* `preprocess_data.py`: Converts the dataset to numpy format for training.
* `evaluation_cases.txt`: A list of cases used for evaluation - a fixed split of the whole dataset.
* `checksum.json`: A list of cases and their checksum for dataset completeness verification.
 
The `data_loading/` folder contains the necessary load data. Its main components are:
* `data_loader.py`: Implements the data loading.
* `pytorch_loader.py`: Implements the data augmentation and iterators.
 
The `model/` folder contains information about the building blocks of U-Net3D and the way they are assembled. Its contents are:
* `layers.py`: Defines the different blocks that are used to assemble U-Net3D.
* `losses.py`: Defines the different losses used during training and evaluation.
* `unet3d.py`: Defines the model architecture using the blocks from the `layers.py` file.

The `runtime/` folder contains scripts with training and inference logic. Its contents are:
* `arguments.py`: Implements the command-line arguments parsing.
* `callbacks.py`: Collection of performance, evaluation, and checkpoint callbacks.
* `distributed_utils.py`: Defines a set of functions used for distributed training.
* `inference.py`: Defines the evaluation loop and sliding window.
* `logging.py`: Defines the MLPerf logger.
* `training.py`: Defines the training loop.

 
# 3. Quality

## Quality metric

The quality metric in this benchmark is mean (composite) DICE score for classes 1 (kidney) and 2 (kidney tumor). 
The metric is reported as `mean_dice` in the code.

## Quality target

The target `mean_dice` is 0.908.

## Evaluation frequency

The evaluation schedule depends on the number of samples processed per epoch. Since the dataset is fairly small, and the
global batch size respectively large, the last batch (padded or dropped) can represent a sizable fraction of the whole dataset.
This implementation assumes that the last batch is always dropped. The evaluation schedule depends on the `samples per epoch` in the following manner:
- for epochs 1 to CEILING(1000*168/`samples per epoch`) - 1: Do not evaluate
- for epochs >= CEILING(1000\*168/`samples per epoch`): Evaluate every CEILING(20\*168/`samples per epoch`) epochs

Two examples:
1. Global batch size = 32:
- `samples per epoch` = 160, since the last batch of 8 is dropped
- evaluation starts at epoch = 1050
- evaluation is run every 21 epochs

2. Global batch size = 128:
- `samples per epoch` = 128, since the last batch of 40 is dropped
- evaluation starts at epoch = 1313
- evaluation is run every 27 epochs

The training should stop at epoch = CEILING(10000\*168/`samples per epoch`). If the model has not converged by that 
epoch the run is considered as non-converged.

## Evaluation thoroughness

The validation dataset is composed of 42 volumes. They were pre-selected, and their IDs are stored in the `evaluation_cases.txt` file.
A valid score is obtained as an average `mean_dice` score across the whole 42 volumes. Please mind that a multi-worker training in popular frameworks is using so-called samplers to shard the data.
Such samplers tend to shard the data equally across all workers. For convenience, this is achieved by either truncating the dataset, so it is divisible by the number of workers,
or the "missing" data is copied. This most likely will influence the final score - a valid evaluation is performed on exactly 42 volumes and each volume's score has a weight of 1/42 of the total sum of the scores.  -->