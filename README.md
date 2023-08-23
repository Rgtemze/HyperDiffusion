# HyperDiffusion
Official code repository of "HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion" @ ICCV 2023

[Paper](https://arxiv.org/abs/2303.17015) | [Project website](https://www.ziyaerkoc.com/hyperdiffusion)

## Method Overview

![Overview image](/static/overview.svg)


## Code Map
### Directories
- **Pointnet_Pointnet2_pytorch**: Includes Pointnet2 definition and weights for 3D FID calculation
- **configs**: Containing training and overfitting configs.
- **diffusion**: Contains all the diffusion logic. Borrowed from [OpenAI](https://github.com/openai/guided-diffusion) 
- **ldm**: Latent diffusion codebase for Voxel baseline. Borrowed from [official LDM repo](https://github.com/CompVis/latent-diffusion)
- **siren**: Modified [SIREN](https://github.com/vsitzmann/siren) codebase. Includes shape overfitting logic.
- **static**: Images for README file
### Generated Directories
- **lightning_checkpoints**: This will be created once you start training for the first time. It will include checkpoints of the diffusion model, the sub-folder names will be the unique name assigned by the Weights & Biases in addition to timestamp.
- **outputs**: Hydra creates this folder to store the configs but we mainly send our outputs to Weights & Biases, so, it's not that special
- **orig_meshes**: Here we put generated weights as .pth and sometimes generated meshes.
### Files
- **augment.py**: Including some augmentation methods, though we don't use them in the main paper
- **dataset.py**: `WeightDataset` and `VoxelDataset` definitions which are `torch.Dataset` descendents. Former one is related to our HyperDiffusion method, while the latter one is for Voxel baseline
- **embedder.py**: Positional encoding definition
- **evaluation_metrics_3d.py**: Methods to calculate MMD, COV and 1-NN from [DPC](https://github.com/luost26/diffusion-point-cloud). Both for 3D and 4D.
- **hd_utils.py**: Many utility methods ranging from rendering to flattening MLP weights
- **hyperdiffusion.py**: Definition of our method, it includes training, testing and validation logics in the form of a Pytorch Lightning module.
- **hyperdiffusion_env.yaml**: Conda environment file (see [Get Started](#get-started) section).
- **main.py**: Entry point of our codebase
- **mlp_models.py**: Definition of ReLU MLPs with positional encoding
- **torchmetrics_fid.py**: Modified torchmetrics fid implementation to calculate 3D-FID
- **transformer.py**: GPT definition from [G.pt paper](https://github.com/wpeebles/G.pt)



## Get Started

We have environment file that you can create a conda environment from. Simply run,

```commandline
conda env create --file hyperdiffusion_env.yaml
conda activate hyper-diffusion
```
To start training, airplane category:
```commandline
python main.py --config-name=plane
```
car category:
```commandline
python main.py --config-name=car
```
chair category:
```commandline
python main.py --config-name=chair
```

We are using [hydra](https://hydra.cc/), you can either specify them from corresponding yaml file or directly modify
the parameters from terminal. For instance, to change the number of epochs:

```commandline
python main.py --config-name=plane epochs=1
```

## Training Plots

We share training plots for better reproducibility. Links take you to Weights & Biases reports.

[Plane](https://api.wandb.ai/links/ziyaer/9korb518) | [Car](https://api.wandb.ai/links/ziyaer2/s528ygbt) | [Chair](https://api.wandb.ai/links/ziyaer2/y9pbdzwh)

## Dependencies

* Python 3.7
* PyTorch 1.13.0
* CUDA 11.7
* Weights & Biases (We heavily rely on it for visualization and monitoring)

For full list please see [hyperdiffusion_env.yml file](/hyperdiffusion_env.yml)