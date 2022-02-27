# MoCo
This is a re-implementation of [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722).  
We referenced to the official PyTorch implementation (https://github.com/facebookresearch/moco) for the re-implementation of MoCo.

Our project is based on CIFAR-10 dataset.

## Requirements
- Python
- PyTorch
- torchvision
- pandas
- matplotlib

## Usage

### Train
Follow the instructions and run the notebook "main_moco.ipynb".
Note the notebook default settings are for Google Colab environments.
You can skip the first and second cells if run locally.

### Experiments  
You can update parameter settings in the fourth cell in the notebook.
Detailed instructions are provided in the notebook.
The file "model_moco.py" contains MoCo and other supporting model classes. You can tweak the model structure from there. However, it is recommended doing so only after you understand MoCo well.

## References
```
@article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
