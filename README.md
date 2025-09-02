#### EuroSys’26 Artifact Evaluation for Paper#1084 MinatoLoader
# 📖 Introduction to MinatoLoader

**Overview:**
MinatoLoader is a general-purpose data loader for PyTorch that accelerates training and improves GPU utilization by eliminating stalls caused by slow preprocessing samples. It continuously prepares data in the background and actively constructs batches by prioritizing fast-to-preprocess samples, while slower samples are handled in parallel and fetched later by the main process. The system is evaluated on servers equipped with NVIDIA V100 and A100 GPUs, demonstrating significant improvements over PyTorch DataLoader, NVIDIA DALI, and Pecan.

**Workflow:**
MinatoLoader operates in three main stages: data loading, preprocessing, and batch construction. Samples are fetched from storage and preprocessed in parallel by CPU workers. A lightweight load balancer applies a per-sample timeout to classify inputs as fast or slow. Fast samples are placed in the fast queue, while those exceeding the timeout are diverted to a temp queue to finish preprocessing in the background; once ready, they move into the slow queue. 
<p align="center"> <img src="speedyloader-diagram.svg" alt="MinatoLoader Workflow" width="600"/> </p>

Each GPU maintains its own batch queue, which assembles training batches from both fast and slow samples, preventing head-of-line blocking. To sustain throughput, a worker scheduler continuously monitors queue occupancy and CPU utilization, dynamically adjusting the number of CPU workers. This design ensures that data preparation overlaps with training, keeping GPUs busy and maximizing training efficiency.

## 2. Execution Environment

### 🖥️ Hardware Specifications
- **Operating System**: Ubuntu 20.04.6 LTS (Focal Fossa), Linux kernel 5.15.0-1066-oracle  
- **CPU**: 2 × Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz (20 cores per socket × 2 sockets × 2 threads = 80 CPUs total)  
- **RAM**: 503 GiB  
- **GPU**: 8 × NVIDIA Tesla V100-SXM2-32GB  
- **Storage**:  446 GB SSD (mounted at `/`)  

### ⚙️ Software Stack
- **NVIDIA Driver**: 560.35.05  
- **CUDA**: 12.6  
- **cuDNN**: 8.9.7  
- **PyTorch**: 2.4.1  
- **Python**: 3.8.10  
- **Docker**: 28.1.1  

## 3. Description of the artifact componenets

This artifact is organized under the `imseg_workload/` directory, which contains one main workload (3D U-Net for image segmentation) and three system implementations. These correspond to the data loading frameworks compared in the paper:  
1. **PyTorch DataLoader** (`imseg_workload/PyTorch`)  
2. **NVIDIA DALI** (`imseg_workload/DALI`)  
3. **MinatoLoader (our system)** (`imseg_workload/Minato`)  

Each system subdirectory follows the same structure:  
- **`data_loading/`** – Implements the data loading phase.  
- **`model/`** – Contains the U-Net3D model components.  
- **`runtime/`** – Implements the training and inference routines.  
- **`main.py`** – Entry point script for running the full pipeline, from data loading and distribution initialization to training.  
- **`run_and_time.sh`** - Running the workload.
- **`run_dali.sh`, `run_minato.sh`, and `run_pytorch.sh`** scripts to run the each workload. 


## Repository Contents

In `imseg_workload/` directory, the most relevant files are:  
- **`checksum.json`** – List of dataset cases and checksums for completeness verification.  
- **`preprocess_data.py`** – Converts the dataset to NumPy format for training.  
- **`requirements.txt`** – Python dependencies for running 3D U-Net.  
- **`Dockerfile`** – Container setup with the required dependencies.  
- **`start_container.sh`** – Script to start the container environment.  
- **`cpu_gpu_usage.sh`** - Outputs a csv file of the average CPU and GPU usage over time.
- **`run_all.sh`** - Run all three systems at once, it accepts the number of GPUs as input. 



Within each system subdirectory (`DALI`, `PyTorch`, `Minato`):  
- **`main.py`** – Main entry point encapsulating the training pipeline.  
- **`evaluation_cases.txt`** – Fixed split of dataset cases used for evaluation.  

### Data Loading Components (`data_loading/`)
- **`data_loader.py`** – Base implementation of the data loading interface.  
- **`pytorch_loader.py`** – Data augmentation and iterators for PyTorch.  
- **`nvidia_daliloader.py`** – Data pipeline implementation for DALI.  
- **`Asynchronous_dataloader.py`** – MinatoLoader implementation with asynchronous logic.  

### Model Components (`model/`)
- **`layers.py`** – Building blocks for assembling U-Net3D.  
- **`losses.py`** – Training and evaluation loss functions.  
- **`unet3d.py`** – U-Net3D model definition built from `layers.py`.  

### Runtime Components (`runtime/`)
- **`arguments.py`** – Command-line argument parsing.  
- **`callbacks.py`** – Callbacks for performance tracking, evaluation, and checkpointing.  
- **`distributed_utils.py`** – Utilities for distributed training.  
- **`inference.py`** – Evaluation loop and sliding-window inference.  
- **`logging.py`** – MLPerf-compatible logging utilities.  
- **`training.py`** – Training loop implementation.  


## 4. Benchmark & Dataset 

This benchmark represents a 3D medical image segmentation task using [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) dataset called [KiTS19](https://github.com/neheller/kits19). The task is carried out using a [U-Net3D](https://arxiv.org/pdf/1606.06650.pdf) model variant based on the [No New-Net](https://arxiv.org/pdf/1809.10483.pdf) paper.The data is stored in the [KiTS19 github repository](https://github.com/neheller/kits19). This code is taken from the [MLCommons Training Image Segmentation Workload](https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch).   

## Steps to download and preprocess the data
1. Clone the MinatoLoader Eurosys  repo
```bash 
git clone git@github.com:Rahm-no/MinatoLoader.git
```
2. Build docker image [This step will take 5 min  ]
```bash 
cd imseg_workload
docker build -t minato:latest .
```
3. Download the data [This step will take 48 min and 27GB in storage ]
   
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

3. Start an interactive session in the container to run preprocessing/training.

    You will need to mount two (or three) directories:

    for raw data (RAW-DATA-DIR)
    for preprocessed data (PREPROCESSED-DATA-DIR)
    (optionally) for results (RESULTS-DIR)


    ```bash 
    mkdir data
    mkdir results
    ./start_container.sh
    ```
 
2. Preprocess the dataset [This step will take 15 min and 29GB of storage].
   
    
    The data preprocessing script is called `preprocess_dataset.py`. All the required hyperparameters are already set. All you need to do is to invoke the script with correct paths:
    ```bash
    python3 preprocess_dataset.py --data_dir /raw_data --results_dir /data
    ```
   
    The script will preprocess each volume and save it as a numpy array at `/data`. It will also display some statistics like the volume shape, mean and stddev of the voxel intensity. Also, it will run a checksum on each file comparing it with the source. This preprocessing step will produce a numpy array for each image (presented by _x) and its corresponding label(presented by _y). 





## 4. Running the Systems

All experiments must be executed **inside the provided Docker container**.  
You can either:  
1. **Run all systems at once** (Option A).
2. **Run each system individually** (Option B) .  

In both cases, the **final step is evaluating results and plotting figures**.

---

### Step 1: Start the container
First, make sure the Docker image has been built using the provided `Dockerfile`.  
Then, launch the container with:

```bash
./start_container.sh
```
This script will start the container and mount the repository. Once inside, you will find yourself in the directory of the artifact: `imseg-workload`.
### Option A: Run All Systems at Once 

To run PyTorch, DALI, and MinatoLoader sequentially with a single command:
``` bash 
./run_all.sh NUM_GPUs
```
* NUM_GPUs = number of GPUs to use (default: 8)

* Example (on a node with 8×V100 GPUs): ``` ./run_all.sh 8 ```

### Option B: Run each system
### Step 2: Choose the system to run 
Navigate into the system directory you want to evaluate. For example: ```cd DALI ```

### Step 3: Run the training script
Each system provides a wrapper script to launch training:

``` bash 
run_SYSTEM.sh NUM_GPUs 
```

Replace SYSTEM with the chosen implementation (pytorch, dali, or minato). Replace NUM_GPUs with the number of GPUs to use (e.g., 2, 4, or 8).
Example: to run MinatoLoader on 8 GPUs: ``` bash run_minato.sh 8```.
With 8 GPUs, running DALI took x seconds, Minato took around x seconds, and PyTorch takes x seconds. 

## 5. Evaluate Results

When you launch a run, the system will automatically:  
- **Train a 3D U-Net model** on the preprocessed dataset,  
- **Log training metrics**, including throughput, CPU/GPU utilization, and total runtime,  
- **Save model checkpoints** under the `ckpts/` directory,  
- **Write results to CSV files**, which include the measured training time. These CSV files can later be used for plotting figures and further analysis.  

For comparability and faster turnaround, we report **time-to-train with 8 GPUs** across all systems.  
However, you can also run with fewer GPUs (e.g., `bash run_pytorch.sh 2`), though the training will naturally take longer.  

After running the experiments, you can produce visual summaries of the results:  

- **Training time comparison**  
   To generate a histogram comparing the training time of the three systems (PyTorch, DALI, and MinatoLoader):  
   ```python3 plot_figure.py```
- **Resource utilization yimeline**
To visualize CPU and GPU utilization over time:
``` python3 plot_usage.py```
These plots provide a clear picture of both performance (time-to-train) and efficiency (hardware usage) across systems.