#### EuroSys’26 Artifact Evaluation for Paper#1084 MinatoLoader
## 📖 Introduction to MinatoLoader

**Overview:**
MinatoLoader is a general-purpose data loader for PyTorch that accelerates training and improves GPU utilization by eliminating stalls caused by slow preprocessing samples. It continuously prepares data in the background and actively constructs batches by prioritizing fast-to-preprocess samples, while slower samples are handled in parallel and fetched later by the main process. The system is evaluated on servers equipped with NVIDIA V100 and A100 GPUs, demonstrating significant improvements over PyTorch DataLoader, NVIDIA DALI, and Pecan.

**Workflow:**
MinatoLoader operates in three main stages: data loading, preprocessing, and batch construction. Samples are fetched from storage and preprocessed in parallel by CPU workers. A lightweight load balancer applies a per-sample timeout to classify inputs as fast or slow. Fast samples are placed in the fast queue, while those exceeding the timeout are diverted to a temp queue to finish preprocessing in the background; once ready, they move into the slow queue. 
<p align="center"> <img src="speedyloader-diagram.svg" alt="MinatoLoader Workflow" width="600"/> </p>

Each GPU maintains its own batch queue, which assembles training batches from both fast and slow samples, preventing head-of-line blocking. To sustain throughput, a worker scheduler continuously monitors queue occupancy and CPU utilization, dynamically adjusting the number of CPU workers. This design ensures that data preparation overlaps with training, keeping GPUs busy and maximizing training efficiency.

##  Execution Environment

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

💡 **Note:** The artifact experiments were executed on the setup described above. In the paper, we additionally report results on **NVIDIA A100 GPUs** to broaden the evaluation.  

## 📦 Artifact Components

This artifact is organized around one main workload (**3D-UNet for image segmentation**) and three system implementations. These correspond to the data loading frameworks compared in the paper:  
1. **PyTorch DataLoader** (`PyTorch/`)  
2. **NVIDIA DALI** (`DALI/`)  
3. **MinatoLoader (our system)** (`Minato/`)  

Each system subdirectory (`PyTorch`, `DALI`, `Minato`) follows a similar structure:  
- **`main.py`** – Entry point script for the full pipeline (data loading, data distribution, training).  
- **`run_and_time.sh`** – Script to run the workload.  
- **System-specific launcher** (`run_pytorch.sh`, `run_dali.sh`, `run_minato.sh`) – Scripts to run each system individually.  
- **`evaluation_cases.txt`** – Fixed split of dataset cases used for evaluation.  
- Subdirectories:  
  - **`data_loading/`** – Data loading implementations.  
  - **`model/`** – 3D-UNet model components.  
  - **`runtime/`** – Training, inference, and distributed execution routines.  

### Repository Contents

At the repository root, the most relevant files are:  
- **`preprocessing_data/`** – Dataset preprocessing utilities (e.g., `preprocess_dataset.py` to convert the dataset to NumPy format). 
- **`results/`** – Stores output results, organized into subdirectories by system (PyTorch, DALI, Minato).  
- **`figures/`** – Contains plots generated from the analysis scripts.  
- **`scripts/`** – Helper scripts:  
  - **`cpu_gpu_usage.sh`** – Logs average CPU and GPU usage to CSV.  
  - **`run_all.sh`** – Runs all three systems at once; accepts the number of GPUs as input.  
  - **`plot_figure.py`**, and **`plot_usage.py`** - Plotting utilities for reproducing figures.  
- **`checksum.json`** – Dataset case list and checksums for completeness verification.  
- **`requirements.txt`** – Python dependencies for running 3D-UNet workload.  
- **`Dockerfile`** – Container setup with all required dependencies.  
- **`start_container.sh`** – Script to launch the container environment.  
- **`speedyloader-diagram.svg`** – Workflow diagram of MinatoLoader.  


#### Data Loading Components (`data_loading/`)
- **`data_loader.py`** – Base implementation of the data loading interface.  
- **`pytorch_loader.py`** – Data augmentation and iterators for PyTorch.  
- **`nvidia_daliloader.py`** – Data pipeline implementation for DALI.  
- **`asynchronous_dataloader.py`** – MinatoLoader’s implementation.  


## 🧪 Benchmark & Dataset

This benchmark represents a 3D medical image segmentation task using the [2019 Kidney Tumor Segmentation Challenge (KiTS19)](https://kits19.grand-challenge.org/) dataset. The task is carried out with a [U-Net3D](https://arxiv.org/pdf/1606.06650.pdf) model variant inspired by the [No New-Net](https://arxiv.org/pdf/1809.10483.pdf) paper. The KiTS19 dataset is hosted in the [official GitHub repository](https://github.com/neheller/kits19), and the baseline code originates from the [MLCommons Training Image Segmentation Workload](https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch).

💡 **Note:** While KiTS19 serves as the primary benchmark in this artifact, we also evaluated our system on other datasets, including **COCO** ([https://cocodataset.org/#home]) for object detection and **LibriSpeech** ([https://www.openslr.org/12]) for speech recognition, to validate its generality across different workloads.


### Steps to download and preprocess the data

1. Clone the MinatoLoader Eurosys  repo
```bash 
git clone https://github.com/Rahm-no/MinatoLoader.git
cd MinatoLoader

```

2. Build docker image [⚠️ This step will take 5 min  ]
```bash 
docker build -t minato:latest .
```

💡 **Note:** All dependencies (system libraries, Python packages, CUDA/cuDNN, PyTorch, DALI, etc.) are automatically installed inside the Docker image.  
No manual setup is required beyond building the container.  

3. Download the data [⚠️ This step will take 48 min and 27GB in storage ]
   
    To download the data, please follow the instructions:
    ```bash
    cd raw-data-dir/kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    ```
    This will download the original, non-interpolated data to `raw-data-dir/kits19/data`

3. Start an interactive session in the container to run preprocessing of the dataset and model training. 

    ```bash 
    cd ../.. # return to root directory 
    mkdir data
    mkdir results
    ./start_container.sh
    ```
 
2. Preprocess the dataset [This step will take ~13 min and 29GB of storage].
   
    
    The data preprocessing script is called `preprocess_dataset.py`. All the required hyperparameters are already set. All you need to do is invoke the script with the correct paths:
    ```bash
    python3 preprocess_dataset.py --data_dir /raw_data --results_dir /data
    ```
   
    The script will preprocess each volume and save it as a numpy array at `/data`. It will also display some statistics like the volume shape, mean, and stddev of the voxel intensity. Also, it will run a checksum on each file, comparing it with the source. This preprocessing step will produce a numpy array for each image (presented by _x) and its corresponding label(presented by _y). 

## 🚀 Running the Systems

All experiments must be executed **inside the provided Docker container**.  
The general workflow is:

1. **Start the container**  (if not already started)
2. **Run experiments** (all systems at once, or each individually)  
3. **Evaluate results and generate figures**  

---

### Step 1: Start the Container (if not already started)
First, ensure the Docker image has been built (see above).

⚠️ You may have already started the container (steps 1-4 of ## Benchmark & Dataset). If you are following the steps from the beginning, you are probably already inside the container (in this case, directly continue to step 2!)
  
Then launch the container with:

```bash
./start_container.sh
```
This script mounts the repository and places you inside the root directory.


### Step 2: Run Experiments

You have two options:

#### 🔹 Option A: Run All Systems

Run PyTorch, DALI, and MinatoLoader sequentially with one command:

``` bash 
./scripts/run_all.sh #NUM_GPUs
```
* #NUM_GPUs = number of GPUs to use (default: 8)

* Example (on a node with 8×V100 GPUs): ``` ./scripts/run_all.sh 8 ```

👀 **Expected output:**  The program should print the following message at the bottom of the output, indicating successful execution of all stages.
``` bash
✅ All systems finished. Combined results are in /workspace/unet3d/results_allsystems.csv
```
#### 🔹 Option B: Run Each System Individually

Navigate into the chosen system’s directory (PyTorch/, DALI/, or Minato/) and launch training with:


``` bash 
./run_SYSTEM.sh NUM_GPUs 
```

Replace SYSTEM with the chosen implementation (pytorch, dali, or minato). Replace NUM_GPUs with the number of GPUs to use (e.g., 2, 4, or 8).
Example: to run MinatoLoader on 8 GPUs: ``` ./run_minato.sh 8```.


👀 **Expected output:**  The program should print the following message at the bottom of the output, indicating successful execution of all stages. X is the system name, it can be either (Minato, Pytorch or DALI).
``` bash
✅ Training for system 'X' completed successfully!
```
### Step 3: Evaluate Results

#### 1. Automatic Outputs
After each run, the system will automatically:
- Train a **3D-UNet** model on the preprocessed dataset  
- Log training metrics (throughput, CPU/GPU utilization, total runtime) to CSV files  
- Save checkpoints under `ckpts/`  
- Append training time results into `results/results_allsystems.csv`  

#### 2. Reference training time
For comparability, we report training time results on **8 GPUs for 10 epochs**:

- ⚡ **PyTorch**: ~210 s  
- 🚀 **DALI**: ~151 s  
- 🌀 **MinatoLoader**: ~81 s  

#### 3. Visualization
You can generate figures to summarize both performance and efficiency.

* 📉 Training Time Comparison

Generate a histogram comparing runtimes across PyTorch, DALI, and Minato:

```bash
python3 scripts/plot_figure.py
```

* 📈 Resource Utilization Timeline

Visualize CPU/GPU utilization over time:

```bash
python3 scripts/plot_usage.py
```
 ✅  Figures will be generated under `figures/`.  These figures demonstrate both:  
- **Performance** → overall training time across the evaluated systems.  
- **Efficiency** → CPU/GPU utilization during training.





