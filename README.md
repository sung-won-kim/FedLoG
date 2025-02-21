# Subgraph Federated Learning for Local Generalizatioin (FedLoG)

The official source code for [**Subgraph Federated Learning for Local Generalization**](https://openreview.net/forum?id=cH65nS5sOz) at ICLR 2025 (Oral).

## Abstract 
Federated Learning (FL) on graphs enables collaborative model training to enhance performance without compromising the privacy of each client. However, previous methods often overlook the mutable nature of graph data, which frequently introduces new nodes and leads to shifts in label distribution. Unlike prior methods that struggle to generalize to unseen nodes with diverse label distributions, our proposed method FedLoG effectively addresses this issue by alleviating the problem of local overfitting. Our model generates global synthetic data by condensing the reliable information from each class representation and its structural information across clients. Using these synthetic data as a training set, we alleviate the local overfitting problem by adaptively generalizing the absent knowledge within each local dataset. This enhances the generalization capabilities of local models, enabling them to handle unseen data effectively. Our model outperforms the baselines in proposed experimental settings, which are designed to measure generalization power to unseen data in practical scenarios.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/90c857a3-ab58-471f-9d43-cadd82bcb543" />

## Requirements
- python=3.10
- pytorch=2.0.1
- torch-geometric=2.4.0

## Datasets
You can download the datasets using the following anonymous link:  
[Download Dataset](https://drive.google.com/file/d/12u40AJMXeeplxfSeOuhU29rv8YtBzWMl/view?usp=share_link)

After extracting `data.zip`, put the `data` folder in the same directory as the `main.py` file.

## How to run
```
python main.py --dataset cora --n_silos 3 --unseen_setting closeset
```
### Flags
`dataset` : cora, citeseer, pubmed, photo, computers  
`n_silos` : 3, 5, 10  
`unseen_setting` : closeset, openset  

### Unseen Settings
<img width="500" alt="image" src="https://github.com/user-attachments/assets/737305da-7f97-45fa-850a-0ecdf7203b8f" style="display: block; margin: 0 auto;"/>

### Citation  

```BibTex
@inproceedings{
kim2025subgraph,
title={Subgraph Federated Learning for Local Generalization},
author={Sungwon Kim and Yoonho Lee and Yunhak Oh and Namkyeong Lee and Sukwon Yun and Junseok Lee and Sein Kim and Carl Yang and Chanyoung Park},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=cH65nS5sOz}
}
```

