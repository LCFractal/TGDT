# Efficient Token-Guided Image-Text Retrieval with Consistent Multimodal Contrastive Training

This repo is built on top of [VSE++](https://github.com/fartashf/vsepp) and [TERAN](https://github.com/mesnico/TERAN).

## Setup

Setup python environment using conda:
```
conda env create --file environment.yml
conda activate gls
export PYTHONPATH=.
```

## Get the data
1. Download and extract the data folder, containing annotations, the splits by Karpathy et al. and ROUGEL - SPICE precomputed relevances for both COCO and Flickr30K datasets:

```
wget http://datino.isti.cnr.it/teran/data.tar
tar -xvf data.tar
```

2. Download the bottom-up features for both COCO and Flickr30K. We use the code by [Anderson et al.](https://github.com/peteanderson80/bottom-up-attention) for extracting them.
The following command extracts them under `data/coco/` and `data/f30k/`. If you prefer another location, be sure to adjust the configuration file accordingly.
```
# for MS-COCO
wget http://datino.isti.cnr.it/teran/features_36_coco.tar
tar -xvf features_36_coco.tar -C data/coco

# for Flickr30k
wget http://datino.isti.cnr.it/teran/features_36_f30k.tar
tar -xvf features_36_f30k.tar -C data/f30k
```

## Evaluate
Download and extract our [pre-trained](https://drive.google.com/file/d/1YYodEraINdJOUg9kSZ9te9qu9Q96MeQU/view?usp=sharing) models.

Then, issue the following commands for evaluating a given model.
```
# F30K
python3 test.py runs/f30k_m0.3/model_best_rsum.pth.tar --config configs/f30k_global.yaml
python3 test.py runs/f30k_m0.3/model_best_rsum.pth.tar --config configs/f30k_local.yaml
python3 test_gl.py runs/f30k_m0.3/model_best_rsum.pth.tar --config configs/f30k_local.yaml

# COCO
python3 test.py runs/coco_m0.3/model_best_rsum.pth.tar --config configs/coco_global.yaml
python3 test.py runs/coco_m0.3/model_best_rsum.pth.tar --config configs/coco_local.yaml
python3 test_gl.py runs/coco_m0.3/model_best_rsum.pth.tar --config configs/coco_local.yaml
```

## Train
In order to train the model using a given configuration, issue the following command:
```
python3 train.py --config configs/f30k_all.yaml --logger_name runs/f30k_m0.3
python3 train.py --config configs/coco_all.yaml --logger_name runs/coco_m0.3
```

# Citation

Please cite this work if you find it useful:.

```
@article{liu2023efficient,
  title={Efficient Token-Guided Image-Text Retrieval with Consistent Multimodal Contrastive Training},
  author={Liu, Chong and Zhang, Yuqi and Wang, Hongsong and Chen, Weihua and Wang, Fan and Huang, Yan and Shen, Yi-Dong and Wang, Liang},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}
```

