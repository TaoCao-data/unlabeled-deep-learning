# MixMatch
This is a re-implementation of [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249).  
We referenced to the official Tensorflow implementation (https://github.com/google-research/mixmatch), an unofficial PyTorch implementation (https://github.com/YU1ut/MixMatch-pytorch) for the re-implementation of MixMatch.  
For the experiments with ResNet-32 and Pseudo Label, we referenced to (https://github.com/akamaster/pytorch_resnet_cifar10) and (https://github.com/perrying/realistic-ssl-evaluation-pytorch).

Our project is based on CIFAR-10 dataset.

## Requirements
- Python
- PyTorch
- torchvision
- numpy

## Usage

#### Train
Run "train_MixMatch.py" for the results.  
The default number of epochs is 10 to reduce computing time.  
The original implementation uses 1,024 epochs.

#### Experiments  
Run code:  
"train_MixMatch_T1.py" for T=1  
"train_MixMatch_k1.py" for K=1  
"train_MixMatch_k3.py" for K=3  
"train_MixMatch_k4.py" for K=4  
"train_MixMatch_noMixUp.py" for w/o MixUp   

#### Benchmark
Run code "train_resnet.py" for results of the supervised model.   
The default number of epochs is 50 to reduce computing time.  
Run code "build_dataset.py" and "train_pl.py" for the results of pseudo label model.  

## Results (Accuracy)
| #Labels | 250 | 500 | 1000 | 2000| 4000 |
|:---|:---:|:---:|:---:|:---:|:---:|
|Paper | 88.92 ± 0.87 | 90.35 ± 0.94 | 92.25 ± 0.32| 92.97 ± 0.15 |93.76 ± 0.06|
|This code | 75 | 82 | 86 | 89 | 91 |

The results are based on 100 epochs, while the results in the paper are based on 1,024 epochs.  

## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
