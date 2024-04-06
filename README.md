# MRHGNN: Enhanced Multimodal Relation Hypergraph Neural Network for Synergistic Drug Combination Forecasting


## Overview

This repository contains Python codes and datasets necessary to run the MRHGNN model. MRHGNN is a novel framework for predicting synergistic drug combinations. Specifically, we design a dual-channel architecture to capture the physicochemical attributes of drugs and their interactive synergies, thereby facilitating the generation of multimodal drug representations. To obtain comprehensive representations of drugs, we utilize an attention mechanism to explore complementarity among multimodal drug embeddings. Additionally, the unified framework jointly learns the primary and self-supervised learning tasks, fostering a robust predictive capability. Please take a look at our paper for more details on the method.

<p align="center">

<img src="https://github.com/Redamancy-CX330/MRHGNN/blob/main/Overall%20Framework.png" align="center">

</p>


## Install the Environment

### OS Requirements

The package development version is tested on _Linux_ (Ubuntu 20.04) operating systems with CUDA 11.7.

### Python Dependencies

MRHGNN is tested under ``Python == 3.8.18``. 

We provide a txt file containing the necessary packages for MRHGNN. All the required basic packages can be installed using the following command:

```
pip install -r requirements.txt
```


## Usage

To train and evaluate the model, you could run the following command.

- O'Neil Dataset

```bash
python MRHGNN.py --dataset 'ONEIL' --threshold 30 --alpha 0.4 --learning_rate 1e-3 --weight_decay 1e-6 --epochs 1500
```

- NCI-ALMANAC Dataset

```bash
python MRHGNN.py --dataset 'ALMANAC' --threshold 10 --alpha 0.6 --learning_rate 1e-3 --weight_decay 1e-6 --epochs 1500
```

## The division and application of the drug synergy dataset

Let's offer a comprehensive overview of how the training process entails the division and application of the drug synergy dataset.

- Firstly, a drug synergy dataset can be viewed as tabular data, where each row represents a quadruplet $(drug\ i, drug\ j, cell\ line\ c, label\ l)$, with $l \in \{0,1\}$ (where 0 denotes an antagonistic effect, while 1 denotes a synergistic effect). 
- Then, we split the drug synergy dataset into training, validation, and testing data according to the following procedure.

---
Input: A drug synergy dataset $D$ where each sample is a quadruplet $(i, j, c, l)$, the cell line set $C$, split ratio $(8:1:1)$.

Output: train data $D_{train}$, validation data $D_{valid}$, test data $D_{test}$.

1. Positive sample set $P \leftarrow\varnothing$;
2. Negative sample set $N \leftarrow\varnothing$;
3. for each sample $(i, j, c, l) \in D$ do
4. $~~~~$ if $l==1$ then
5. $~~~~~~~~~P \leftarrow P\cup$ { $(i, j, c,1)$ };
6. $~~~~$ else
7. $~~~~~~~~~N \leftarrow N\cup$ { $(i, j, c,0)$ };
8. $~~~~$ end if
9. end for
10. for each $r \in C$ do
11. $~~~~$ obtain the set $P_r =$ { $(i, j, c,1) \mid  (i, j, c,1) \in P, c = r$ } and $N_r =$ { $(i, j, c,0) \mid (i, j, c,0) \in N, c = r$ };
12. $~~~~$ Calculate the cardinality of sets $P_r$ and $N_r$ as $|P_r|$ and $|N_r|$, respectively;
13. $~~~~$ Randomly divide set $P_r$ according to the split ratio of (8:1:1), and then incorporate them into $D_{train}$, $D_{valid}$, and $D_{test}$, respectively;
14. $~~~~$ Randomly divide set $N_r$ based on the split sizes of $(|N_r|-0.2\times|P_r|, 0.1\times|P_r|, 0.1\times|P_r|)$, and then add them into $D_{train}$, $D_{valid}$, and $D_{test}$, respectively;
15. end for
16. Return $D_{train}$, $D_{valid}$, and $D_{test}$.
---
- We utilize the train data $D_{train}$ to construct the drug combination hypergraph.
- After conducting message passing and aggregation on the drug combination hypergraph $\mathcal{G}_1$ and interaction hypergraph $\mathcal{G}_2$ using RHGNN, we acquired the final embedding $z^*$ of the drug.
- For the training set $D_{train}$, we compute the predicted value $p_{ij}^c$ for each quadruplet $(drug\ i, drug\ j, cell\ line\ c, label\ l)$, then calculate the loss function $\mathcal{L}_ {p}$ to facilitate gradient backpropagation for updating the model parameters. However, for the validation set $D_{valid}$ and test set $D_{test}$, we solely compute the predicted values $p_{ij}^c$ without calculating the loss function $\mathcal{L}_p$ for gradient backpropagation.

## Citation

Please kindly cite this paper if you find it useful for your research. Thanks!
