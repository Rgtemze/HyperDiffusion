'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''
from functools import partial
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import wandb
from omegaconf import DictConfig, open_dict

import copy
# Enable import from parent package
import sys
import os

from scipy.spatial.transform import Rotation
from torch import nn

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ) )))
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ) ))))
from main import render_mesh
from mlp_models import MLP3D
from siren.experiment_scripts.test_sdf import SDFDecoder

from siren import sdf_meshing
from scipy.spatial.transform import Rotation
import numpy as np

from siren import dataio, utils, training, loss_functions
from torch.utils.data import DataLoader
torch.set_num_threads(8)

def get_model(cfg):
    if cfg.model_type == "mlp_3d":
        model = MLP3D(**cfg.mlp_config)
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print('Total number of parameters: %d' % nparameters)

    return model
@hydra.main(version_base=None, config_path="../../overfitting_configs", config_name="overfit_config")
def main(cfg: DictConfig):
    wandb.init(project="hyperdiffusion_overfitting", dir=cfg.wandb_dir, config=dict(cfg), mode="online")
    first_state_dict = None
    if cfg.strategy == "same_init":
        first_state_dict = get_model(cfg).state_dict()
    x_0s = []
    with open_dict(cfg):
        cfg.mlp_config.output_type = cfg.output_type
    curr_lr = cfg.lr
    root_path = os.path.join(cfg.logging_root, cfg.exp_name)
    mesh_jitter = cfg.mesh_jitter
    multip_cfg = cfg.multi_process
    files = [file for file in os.listdir(cfg.dataset_folder) if file not in ["train_split.lst", "test_split.lst", "val_split.lst"]]
    if multip_cfg.enabled:
        if multip_cfg.ignore_first:
            files = files[1:] # Ignoring the first one
        count = len(files)
        per_proc_count = count // multip_cfg.n_of_parts
        start_index = multip_cfg.part_id * per_proc_count
        end_index = min(count, start_index + per_proc_count)
        files = files[start_index:end_index]
        if cfg.strategy == "first_weights":
            first_state_dict = torch.load(os.path.join(root_path, multip_cfg.first_weights_name))
        print(f"Proc {multip_cfg.part_id} is responsible between {start_index} -> {end_index}")
    lengths = []
    names = []
    train_object_names = np.genfromtxt(os.path.join(cfg.dataset_folder, "train_split.lst"), dtype="str")
    train_object_names = set(train_object_names)
    for i, file in enumerate(files):
        for j in range(10 if mesh_jitter and i > 0 else 1):

            # Quick workaround to rename from obj to off
            # if file.endswith(".obj"):
            #     file = file[:-3] + "off"

            if not (file in train_object_names):
                print(f"File {file} not in train_split")
                continue

            filename = file.split(".")[0]
            filename = f"{filename}_jitter_{j}"

            if cfg.input_type == "audio":
                audio_dataset = dataio.AudioFile(filename=os.path.join(cfg.dataset_folder, file))
                coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)
                dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
            else:
                sdf_dataset = dataio.PointCloud(os.path.join(cfg.dataset_folder, file),
                                                on_surface_points=cfg.batch_size,
                                                is_mesh=True,
                                                output_type=cfg.output_type,
                                                out_act=cfg.out_act,
                                                n_points=cfg.n_points,
                                                cfg=cfg)
                dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
            if cfg.strategy == "save_pc":
                continue
            elif cfg.strategy == "diagnose":
                lengths.append(len(sdf_dataset.coords))
                names.append(file)
                continue

            # Define the model.
            model = get_model(cfg).cuda()
            # obj: trimesh.Trimesh = trimesh.load(os.path.join(cfg.dataset_folder, file))
            # # obj.show()
            # from trimesh.voxel import creation as vox_creation
            # vox_size = 0.0015
            # vox: trimesh.voxel.VoxelGrid = vox_creation.voxelize(obj, pitch=vox_size)
            # vox.as_boxes().export(os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply", "vox_input.ply"))
            # resolution = 512
            # grid = torch.full(vox.shape, 0).float()
            # print(vox)
            # indices = vox.sparse_indices
            # grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            #
            # v, f = mcubes.marching_cubes(mcubes.smooth(grid.numpy()), 0)
            # mcubes.export_obj(v, f, os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply", "deneme_mesh.obj"))
            # sdf_meshing.convert_sdf_samples_to_ply(grid, [-1, -1, -1], vox_size * 2, os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply", "vox_deneme.ply"), 0.5)
            # continue
            # Define the loss
            loss_fn = loss_functions.sdf
            if cfg.output_type == "occ":
                loss_fn = loss_functions.occ_tanh if cfg.out_act == "tanh" else loss_functions.occ_sigmoid
            if cfg.input_type == "audio":
                loss_fn = loss_functions.function_mse
            loss_fn = partial(loss_fn, cfg=cfg)
            if cfg.strategy == "first_weights_kl" and first_state_dict is not None:
                loss_fn = partial(loss_fn, first_state_dict=first_state_dict)
            summary_fn = utils.wandb_audio_summary if cfg.input_type == "audio" else utils.wandb_sdf_summary

            filename = f"{cfg.output_type}_{filename}"
            checkpoint_path = os.path.join(root_path, f"{filename}_model_final.pth")
            if os.path.exists(checkpoint_path):
                print("Checkpoint exists:", checkpoint_path)
                continue
            if cfg.strategy == "remove_bad":
                model.load_state_dict(torch.load(checkpoint_path))
                model.eval()
                with torch.no_grad():
                    (model_input, gt) = next(iter(dataloader))
                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}
                    model_output = model(model_input)
                    loss = loss_fn(model_output, gt, model)
                if loss['occupancy'] > 0.5:
                    print('Outlier:', loss)
                continue
            if cfg.strategy == "continue":
                if not os.path.exists(checkpoint_path):
                    continue
                model.load_state_dict(torch.load(checkpoint_path))
            elif first_state_dict is not None and cfg.strategy != "random" and cfg.strategy != "first_weights_kl":
                print("loaded")
                model.load_state_dict(first_state_dict)

            training.train(model=model, train_dataloader=dataloader, epochs=cfg.epochs, lr=curr_lr,
                           steps_til_summary=cfg.steps_til_summary, epochs_til_checkpoint=cfg.epochs_til_ckpt,
                           model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
                           clip_grad=cfg.clip_grad, wandb=wandb, filename=filename, cfg=cfg)
            if i == 0 and first_state_dict is None and (cfg.strategy == "first_weights" or cfg.strategy == "first_weights_kl") and not multip_cfg.enabled:
                first_state_dict = model.state_dict()
                # if cfg.output_type == "occ" and cfg.strategy == "first_weights":
                #     curr_lr = curr_lr * 5
                print(curr_lr)
            state_dict = model.state_dict()
            weights = []
            # target_shape = Config.get("target_shape")
            # curr_weights = Config.get("curr_weights")
            # target = np.prod(target_shape)
            for weight in state_dict:
                weights.append(state_dict[weight].flatten().cpu())
            # weights.append(torch.zeros(target - curr_weights))  # TODO: I don't know if zero is the best value
            weights = torch.hstack(weights)
            x_0s.append(weights)
            tmp = torch.stack(x_0s)
            var = torch.var(tmp, dim=0)
            print(var.shape, var.mean().item(), var.std().item(), var.min().item(), var.max().item())
            print(var.shape, torch.var(tmp))

            if i < 5:
                sdf_decoder = SDFDecoder(cfg.model_type, checkpoint_path, "nerf" if cfg.model_type == "nerf" else "mlp", cfg)
                os.makedirs(os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply"), exist_ok=True)
                if cfg.mlp_config.move:
                    imgs = []
                    for time_val in range(sdf_dataset.total_time):
                        vertices, faces, _ = sdf_meshing.create_mesh(sdf_decoder, os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply", filename + "_" + str(time_val)), N=256,
                                                level=0.5 if cfg.output_type == "occ" and cfg.out_act == "sigmoid" else 0, time_val=time_val)
                        rot_matrix = Rotation.from_euler("zyx", [45, 180, 90], degrees=True)
                        tmp = copy.deepcopy(faces[:, 1])
                        faces[:, 1] = faces[:, 2]
                        faces[:, 2] = tmp
                        obj = trimesh.Trimesh(rot_matrix.apply(vertices), faces)
                        img, _ = render_mesh(obj)
                        imgs.append(img)
                    imgs = np.array(imgs)
                    imgs = np.transpose(imgs, axes=(0, 3, 1, 2))
                    wandb.log({"animation": wandb.Video(imgs, fps=16)})
                else:
                    sdf_meshing.create_mesh(sdf_decoder, os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply", filename), N=256, level=0 if cfg.output_type == "occ" and cfg.out_act == "sigmoid" else 0)
                # break
    # lengths = np.array(lengths)
    # names = np.array(names)
    # print(names)
    # print(np.histogram(lengths, bins=50))
    # for size in range(100, 10000, 100):
    #     idx = lengths - 200000 < size
    #     print(f'Size({size}):', len(lengths[idx]))
    #     print('Names', names[idx])
    # print()
if __name__ == "__main__":
    main()