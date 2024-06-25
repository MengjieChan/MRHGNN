# MRHGNN: Enhanced Multimodal Relational Hypergraph Neural Network for Synergistic Drug Combination Forecasting


## Overview

This repository contains Python codes and datasets necessary to run the MRHGNN model. MRHGNN is a novel framework for predicting synergistic drug combinations. Specifically, we design a dual-channel architecture to capture the physicochemical attributes of drugs and their interactive synergies, thereby facilitating the generation of multimodal drug representations. To obtain comprehensive representations of drugs, we utilize an attention mechanism to explore complementarity among multimodal drug embeddings. Additionally, the unified framework jointly learns the primary and self-supervised learning tasks, fostering a robust predictive capability. Please take a look at our paper for more details on the method.

<p align="center">

<img src="https://github.com/Redamancy-CX330/MRHGNN/blob/main/Overall%20Framework.png" align="center">

</p>


## Install the Environment

### OS Requirements

The package development version is tested on _Linux_ (Ubuntu 20.04) operating systems with CUDA 11.7.

### Python Dependencies

MRHGNN is tested under ``Python == 3.10.14``. 

We provide a txt file containing the necessary packages for MRHGNN. All the required basic packages can be installed using the following command:

```
pip install -r requirements.txt
```


## Usage

To train and evaluate the model, you could run the following command.

- O'Neil Dataset

```bash
python MRHGNN.py --dataset 'ONEIL' --threshold 30 --alpha 0.2 --mask_ratio 0.2 --learning_rate 1e-2 --weight_decay 1e-4 --epochs 2000
```

- NCI-ALMANAC Dataset

```bash
python MRHGNN.py --dataset 'ALMANAC' --threshold 10 --alpha 0.3 --mask_ratio 0.3 --learning_rate 1e-2 --weight_decay 1e-4 --epochs 2000
```

## Citation

Please kindly cite this paper if you find it useful for your research. Thanks!
