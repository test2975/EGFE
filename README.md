# EFGE

Pytorch implementation for EGFE: End-to-end Grouping of Fragmented Elements in GUI Designs with Multimodal Learning.

## Requirements

```sh
pip install -r requirements.txt
```

## Usage

This is the Pytorch implementation of EGFE. It has been trained and tested on Linux (Ubuntu20 + Cuda 11.6 + Python 3.9 + Pytorch 1.13 + 4 * NVIDIA GeForce RTX 3090 GPU),
and it can also work on Windows.

## Getting Started

**Clone this repo:**

`git clone git@github.com:test2975/EGFE.git`  
`cd EGFE`

## Data Preprocessing

`cd sketch_dataset`

* Generate design prototypes(*.sketch) to method format
`python scripts/convert_sketch.py --input [sketch_folder] --output ./data`
* Generate train and test format index files
`python scripts/convert_config.py [config.json file path] ./data/output`
* Since the scripts use [sketchtool](https://developer.sketch.com/cli/export-assets), it has to run on macOS.

## Data preparation

* Contact the authours of this paper to request the training dataset, and modified the path of `train_index_json` and `test_index_json` in `./sketch_model/configs/config.py`.

* Download the [testing dataset](https://zenodo.org/record/8022996), and have it in the 'dataset' folder.

## To Train

* Start to train with

```sh
torchrun --nnodes 1 --nproc_per_node 4  main_ddp.py --batch_size 10 --lr 5e-4
```

## To Test

* Download the pre-trained [EGFE](https://zenodo.org/record/8022996) model, and put it in the `'./work_dir'` folder.
* Start to test with

```sh
torchrun --nnodes 1 --nproc_per_node 4  main_ddp.py --evaluate --resume ./work_dir/set-wei-05-0849/checkpoints/latest.pth --batch_size 40
```
