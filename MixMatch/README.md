# MixMatch
This is a re-implementation of [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249) for class project in CS7643 Deep Learning.  
We referenced to the official Tensorflow implementation (https://github.com/google-research/mixmatch), an unofficial PyTorch implementation (https://github.com/YU1ut/MixMatch-pytorch) for the re-implementation of MixMatch.  
For the experiments with ResNet-32 and Pseudo Label, we referenced to (https://github.com/akamaster/pytorch_resnet_cifar10) and (https://github.com/perrying/realistic-ssl-evaluation-pytorch).

Our project is based on CIFAR-10 data.

## Requirements
- Python
- PyTorch
- torchvision
- numpy

## Usage

### Train and Experiments for MixMatch  
#### Train
Run code "train_MixMatch.py" for the results.  
The default number of epochs is 10 to reduce computing time.  
The original implementation uses 1,024 epochs.

#### Experiments  
Run code:  
"train_MixMatch_T1.py" for T=1  
"train_MixMatch_k1.py" for K=1  
"train_MixMatch_k3.py" for K=3  
"train_MixMatch_k4.py" for K=4  
"train_MixMatch_noMixUp.py" for w/o MixUp    

## Results (Accuracy)
| #Labels | 250 | 500 | 1000 | 2000| 4000 |
|:---|:---:|:---:|:---:|:---:|:---:|
|Paper | 88.92 ± 0.87 | 90.35 ± 0.94 | 92.25 ± 0.32| 92.97 ± 0.15 |93.76 ± 0.06|
|This code | 88.71 | 88.96 | 90.52 | 92.23 | 93.52 |

(Results of this code were evaluated on 1 run. Results of 5 runs with different seeds will be updated later. )

## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
