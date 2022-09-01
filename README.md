[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3552674.svg)](https://doi.org/10.5281/zenodo.3552674)


# ScaleDependentCNN
This repository contains all the necessary to reproduce the work "*Scale dependant layer for self-supervised nuclei encoding*" by Naylor Peter, Yao-Hung Hubert Tsai, Marick Laé and Makoto Yamada.


## Requirements
You will need to have an installation of conda and of [nextflow](https://www.nextflow.io/).
For pytorch and torchvision, please be sure to take compatible versions.

## Nextflow specifics
Please modify the `nextflow.config` to meet the specificities of your informatics setup.
We used a SGE cluster provided by RIKEN AIP with CPU and GPU queues. 
Processing the PanNuke dataset requires more memory and computational power then the others, if you are limited in this regard, please remove this dataset from the `benchmark.nf` file.

## Reproduce results in paper
Please run the commands in this order:
``` bash
# setup the conda environnement 
make setup_conda

# download the data
make download_data

# run the main experiment
make experiment

# run the supplementary experiment of MoCo
make moco_experiment

# run the supplementary experiment to see the effect of different augmentations on the test set
make augmentation_experiment:
```


## Hyper parameter-settings
To reproduce the main experiment in the paper please use the following hyper-parameters:
``` python
methods_selection = ["ascending", "descending"]
LAMBDA = [0.00078125, 0.0078125, 0.078125]
LR = [0.0001, 0.001, 0.01, 0.1]
WD = [0, 1e-6, 1e-4, 1e-2, 1.]
FEATURE_DIM = [32, 64, 128]
MB = [32768, 65536]
KS = [3, 5]
epochs = [ "tnbc": 100, "consep": 100, "pannuke": 50, "tnbcpadded": 100, "conseppadded": 100, "pannukepaddded": 50]
bs = [ "tnbc": [128], "consep": [128], "pannuke": [512], "tnbcpadded": [64], "conseppadded": [64], "pannukepadded": [64]]
number_bs = [0] // This parameter is to iterate over the batch size, please modify if you specify many batch size

models = ["ModelSDRN", "ModelSRN"]
opt = ["--inject_size", "--no_size"]
repetition = 20
```


## Cite
Please cite our paper if you use our dataset or code.
```
@article{naylor2022scaledependant,
  title={Scale dependant layer for self-supervised nuclei encoding},
  author={Naylor, Peter and Yao-Hung, Hubert Tsai and Marick, Laé and Makoto, Yamada},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```
