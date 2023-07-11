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

## Data Preparation

* Contact the authours of this paper to request the training dataset, and modified the path of `train_index_json` and `test_index_json` in `./sketch_model/configs/config.py`.

* Download the [testing dataset](https://zenodo.org/record/8022996), and have it in the 'dataset' folder.

## Train Our Model

* Start to train with

```sh
torchrun --nnodes 1 --nproc_per_node 4  main_ddp.py --batch_size 10 --lr 5e-4
```

## Test Our Model

* Download the pre-trained [EGFE](https://zenodo.org/record/8132008) model, and put it in the `'./work_dir'` folder.
* Start to test with

```sh
torchrun --nnodes 1 --nproc_per_node 4  main_ddp.py --evaluate --resume ./work_dir/set-wei-05-0849/checkpoints/latest.pth --batch_size 40
```

## Baselines of UI Fragmented Element Classification

### EfficientNet

* Start to train with

```sh
torchrun --nnodes 1 --nproc_per_node 4  efficient_main.py --batch_size 4 --lr 5e-4
```

* Start to test with

```sh
torchrun --nnodes 1 --nproc_per_node 4  efficient_main.py --evaluate --resume ./work_dir/efficient_net/latest.pth --batch_size 8
```

### Vision Transformer(ViT)

* Start to train with

```sh
torchrun --nnodes 1 --nproc_per_node 4  vit_main.py --batch_size 4 --lr 5e-4
```

* Start to test with

```sh
torchrun --nnodes 1 --nproc_per_node 4  vit_main.py --evaluate --resume ./work_dir/vit/latest.pth --batch_size 8
```

### Swin Transformer

* Start to train with

```sh
torchrun --nnodes 1 --nproc_per_node 4  sw_vit_main.py --batch_size 4 --lr 5e-4
```

* Start to test with

```sh
torchrun --nnodes 1 --nproc_per_node 4  sw_vit_main.py --evaluate --resume ./work_dir/swin/latest.pth --batch_size 8
```

## Baselines of UI Fragmented Elements Grouping

### UILM

Please refer to <https://github.com/zjl12138/UILM/>

### ULDGNN

Please refer to <https://github.com/zjl12138/ULDGNN>

### UIED

We have released a UI component classifier (binary classification) for UIED trained on our dataset. It can be downloaded from [pretrained model](https://zenodo.org/record/8132008).

Please refer to the open-source code <https://github.com/MulongXie/UIED>

## ACKNOWNLEDGES

The implementations of EfficientNet, Vision Transformer, and Swin Transformer are based on the following GitHub Repositories. Thank for the works.

* EfficientNet: <https://github.com/lukemelas/EfficientNet-PyTorch>
* ViT: <https://github.com/lucidrains/vit-pytorch>
* Swin Transformer: <https://github.com/microsoft/Swin-Transformer>

## Model Resources

You can download all the model checkpoints from [pre-trained model.7z](https://zenodo.org/record/8132008).
