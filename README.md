# PathFL Toolchain: Graph Construction (JSON → PyG .pt) and Training/Evaluation (GAT K-fold / Cross-Project)

This repository contains three parts:  
1) **Dynamic Analysis (Instrumentation + Logs → Dynamic Graph JSON)**: instrument a Defects4J project, run tests to produce execution logs, and parse the logs into a dynamic graph (PDDG) in JSON format.  
2) **Graph Builder (JSON → PyG .pt)**: batch-convert the hybrid dependency graph **HPDG** (JSON) into PyTorch Geometric `Data` (`.pt`) datasets, and export `index.csv` for training-time scanning.  
3) **Training & Evaluation (.pt → Metrics)**: perform fault localization evaluation on `.pt` graph samples with **Within-Project K-fold**, **Cross-Project**, or **LOO**, and report metrics such as Top@K / MAR / MFR.

---

## Repository Layout

Recommended layout (dynamic analysis + builder + training in one repo):

```text
.
├── environment.yml                 # Conda environment (trainning env)
├── build_dataset.py                # Graph builder entry (CLI)
├── train_gat.py                    # Training/Evaluation entry (CLI)
├── phdgfl_builder/                 # Graph Builder (JSON → .pt)
├── phdgfl_train/                   # Training & Evaluation (.pt → metrics)
└── daynamic_analy/                 # Dynamic Analysis (instrument → log → dynamic graph JSON)
    ├── mySootProject/              # Java 11: Soot instrumentation project (run via Maven)
    └── dependency_analy/           # Python: parse logs into dynamic dependency graph JSON
```

> Note: the folder name `daynamic_analy` follows the current repository naming.

---

## Environment

### Conda Environment (Recommended)

The repo provides `environment.yml`. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate oathfl
```

> If you need to match CUDA / PyTorch / PyG versions on your machine, adjust the corresponding versions in `environment.yml`.

### Java Environment (Required for Dynamic Analysis)

- Java 11 (required by `daynamic_analy/mySootProject`)

---

# Quick Start

## 1) Download the dataset from GitHub Release

First, download the dataset from the [GitHub Release page](https://github.com/your_username/HPDG-FL/releases/tag/v1.0) for the processed dataset (`.pt` files).

## 2) Within-Project K-fold Training (default)

Once the dataset is downloaded and extracted, run the following command to start training with GAT on the dataset:

```bash
python train_gat.py \
  --processed_roots /path/to/processed_dataset_v4_120/Time \
  --raw_json_roots  /path/to/raw_json_data/Time \
  --eval_protocol kfold \
  --k_folds 4 \
  --epoch_sweep 20,30,40,50,60 \
  --result_dir ./results \
  --result_text_file kfold_results.txt

---

# Part A: Dynamic Analysis (Defects4J Instrumentation + Logs → Dynamic Graph JSON)

This part extracts dynamic information from **Defects4J bugs** during test execution and outputs a dynamic graph JSON, which will later be converted to `.pt` by the Graph Builder.

---

## Directory Description

`daynamic_analy/` contains two components:

- `mySootProject/`: a **Java 11** project that instruments the target class bytecode (Soot pipeline).
- `dependency_analy/`: a Python log parser that reads `/tmp/instrument.log` and produces a dynamic graph JSON.

---

## Step 1: Run Instrumentation (Java 11)

Enter `mySootProject/` and run instrumentation via Maven:

```bash
mvn clean compile exec:java \
  -Dexec.mainClass="com.example.App" \
  -Dexec.args="-target-class com.example.TargetClass"
```

After it finishes, the target class will be replaced by the **instrumented `.class` files** (overwritten/output as the instrumented version; the exact output location depends on your implementation).

> This step requires Java 11 (e.g., `java -version` shows 11).

---

## Step 2: Run Tests on the Target Defects4J Project to Generate Logs

In the selected Defects4J project directory, run the specified test (example uses `-t` to run a particular test case):

```bash
defects4j test -t <TEST_NAME>
```

This will generate the instrumentation log:

- Log path: `/tmp/instrument.log`

---

## Step 3: Parse Logs to Produce Dynamic Graph JSON (Python)

Enter `daynamic_analy/dependency_analy/` (or run the script from anywhere) and execute:

```bash
python dependency_analy.py <project_name> <n>
```

Parameters:

- `<project_name>`: Defects4J project name (e.g., `Lang` / `Math` / `Time` / `Chart` / `Closure` / `Mockito`)
- `<n>`: the `n`-th bug in that project (e.g., `1` means bug 1)

The script reads `/tmp/instrument.log` and outputs the parsed **dynamic graph (PDDG) JSON**, which will be fed into the Graph Builder to generate `.pt`.

---

# Part B: PHDGFL Graph Builder (JSON → PyG .pt)

After obtaining PDDG, you typically combine it with defect-line labels and a static dependency subgraph (which can be generated via Joern or Soot) to construct the hybrid dependency graph **HPDG**.  
This part batch-converts HPDG JSON into PyTorch Geometric `Data` (`.pt`) and exports `index.csv` for training-time scanning.

---

## Overview

Supported features:

- Batch processing for multiple datasets (Closure/Time/Chart/Mockito/Lang/Math) and multiple projects (pid)
- Automatic detection of input path hierarchy:  
  **global root / single-dataset dir / single-project(pid) dir / a directory directly containing JSON files**
- CodeBERT node semantic vectors (can be disabled, or keep code-only)
- Optional node merging by line number (simulating “no path-context splitting”)
- No-Edges ablation: `edge_build_mode=none`
- Export `index.csv` (`dataset/pid/pt_path`)

---

## Input Data Format

Recommended input layout follows the Defects4J style:

```text
dependency_graphs_all_120/
  Closure/
    1/
      xxx.json
      yyy.json
  Time/
    1/
      ...
```

A single JSON should contain at least:

- `nodes`: dict; keys are node IDs; values may include fields like `line`, `code_content`/`code`, `defs`, `uses`, `calls`, `is_defect`, `from_static`, etc.
- `edges`: dict; keys look like `"src-dst"`; values are edge lists (each item contains fields like `type`, `count`)

---

## Quick Start

### 1) Build the full dataset (multi-dataset, multi-project)

```bash
python build_dataset.py /your/path/to/json_file \
  --output_root /your/path/to/pt_file \
  --datasets Closure,Time,Chart,Mockito,Lang,Math \
  --num_workers 2 \
  --edge_build_mode full \
  --feature_mode full
```

### 2) Build a single dataset (e.g., Math)

```bash
python build_dataset.py /your/path/to/json_file/Math \
  --output_root /your/path/to/pt_file
```

### 3) Build a single project pid (e.g., Math/1)

```bash
python build_dataset.py /your/path/to/json_file/Math/1 \
  --output_root /your/path/to/pt_file
```

### 4) Input directory directly contains JSON files (also supported)

```bash
python build_dataset.py /some/folder/with_jsons \
  --output_root /some/output
```

---

## Output

Default save mode is `per_graph`. Example output structure:

```text
output/
  Closure/
    proj_1/
      xxx.pt
      yyy.pt
    proj_2/
      ...
  Time/
    ...
  index.csv
```

`index.csv` fields:

- `dataset`
- `pid`
- `pt_path`

---

## CLI Arguments (Builder)

Common arguments:

- `--datasets`: dataset names (only effective when `input_root` is the global root)
- `--save_mode {per_graph, per_project}`
  - `per_graph`: one `.pt` per JSON
  - `per_project`: one `.pt` per project (contents are `List[Data]`)
- `--num_workers N`: number of multiprocessing workers
- `--batch_size B`: CodeBERT encoding batch size
- `--merge_nodes_by_line {0,1}`: whether to merge nodes by line number
- `--edge_build_mode {full,none}`: whether to build edges (`none` for no-edges ablation)
- `--feature_mode {full,no_code,code_only}`
  - `full`: `code(768) + struct/dataflow/flags/deg = 779` dims
  - `no_code`: do not run CodeBERT; code vectors are all zeros (still keep structural features)
  - `code_only`: only use CodeBERT semantic vectors (`768` dims)
- `--export_index_csv {0,1}`: whether to export the index table
- `--mp_start_method {spawn,fork,forkserver}`: multiprocessing start method (default `spawn`)

---

## Data Fields in Output `.pt`

Each `.pt` is a `torch_geometric.data.Data` object. Common fields:

- `x`: node features (`[N, 779]` or `[N, 768]`)
- `edge_index`: `[2, E]`
- `edge_type`: `[E]` (discrete type id)
- `edge_weight`: `[E]` (`log1p(count) * coef`)
- `y`: `[N]` node labels (binary labels after defect-line expansion, if used)
- `node_line`: `[N]` node line numbers
- `from_static`: `[N]` whether the node is static (1/0)
- `filename`, `identifier`: metadata

---

# Part C: PHDGFL Training & Evaluation (GAT Training and Evaluation)

This part trains and evaluates a fault localization model based on `.pt` datasets produced by the Graph Builder:

- **Within-Project K-fold** (bucketed by bug/pid)
- **Cross-Project** (leave-one-project-out: train on other projects, test on the target project)
- **LOO** (optional: within-project leave-one-out)

Reported metrics (bug-level by default):

- `Top@K` (K ∈ {1,3,5,10})
- `MAR` (Mean Average Rank)
- `MFR` (Mean First Rank)

It also supports:

- **Edge-type ablation** (e.g., only data/cfg/call, etc.)
- **Line-level merging / line-level supervision (MIL pooling)**
- **epoch_sweep**: sweep multiple epochs and write each trial’s results to a text file

---

## Training Input Requirements

The training entry reads:

1) `processed_roots`: the `.pt` roots produced by the Graph Builder (can be multiple dataset paths)  
2) (optional) `raw_json_roots`: original JSON roots used as a fallback to read line numbers when `.pt` lacks `node_line`  
   - If `.pt` already contains `node_line`, JSON is not required in training.

---

## Quick Start

### 1) Within-Project K-fold (default)

```bash
python train_gat.py \
  --processed_roots /your/path/to/pt_file/Time \
  --raw_json_roots  /your/path/to/json_file/Time \
  --eval_protocol kfold \
  --result_dir ./results \
  --result_text_file kfold_results.txt
```

### 2) Cross-Project (leave-one-project-out)

```bash
python train_gat.py \
  --processed_roots /your/path/to/pt_file/Time \
  --raw_json_roots  /your/path/to/json_file/Time \
  --eval_protocol cross_project \
  --result_dir ./results \
  --result_text_file xproj_results.txt
```

### 3) LOO (optional)

```bash
python train_gat.py \
  --processed_roots /your/path/to/pt_file/Math \
  --raw_json_roots  /your/path/to/json_file/Math \
  --eval_protocol loo \
```

---

## CLI Arguments (Training)

Common arguments:

### Data and Protocol

- `--processed_roots`: one or more `.pt` dataset roots
- `--raw_json_roots`: (optional) JSON roots for fallback `node_line` loading
- `--eval_protocol {kfold,cross_project,loo}`
- `--k_folds K`: only effective for kfold
- `--seed S`

### Training Hyperparameters

- `--epochs E`: max epochs (effective when `epoch_sweep` is not used)
- `--epoch_sweep 20,30,40,...`: sweep multiple epochs; each value is treated as a trial (retrain + evaluate)
- `--batch_size B` (recommended `1`: one graph/sample)
- `--lr LR`
- `--weight_decay WD`
- `--dropout P`
- `--heads H`
- `--hidden D` or `--auto_hidden 1`

### Training Strategies

- `--use_line_level_train {0,1}`: enable line-level supervision (MIL)
- `--line_pool {mean,max,softmax}`: line-level pooling
- `--line_pool_tau T`: softmax pooling temperature

### Edge-type Ablations

- `--edge_subset_mode {full,only_data,only_cfg,only_call,data_cfg,no_data,no_cfg,no_call}`

### Evaluation Settings

- `--merge_reduce {max,mean}`
- `--global_use_file_norm {0,1}`
- `--global_norm_type {zscore,minmax,rank}`
- `--global_top_m_per_file M`

### Performance and Acceleration

- `--use_amp {0,1}`
- `--amp_dtype {bf16,fp16}`
- `--num_workers N`
- `--pin_memory {0,1}`
- `--cache_graphs_in_memory {0,1}`
- `--cache_max_items N`
- `--use_tf32 {0,1}`

### Output

- `--result_dir PATH`
- `--result_text_file NAME`
