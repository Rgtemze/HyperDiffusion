import os
from math import ceil, floor
from os.path import join

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from trimesh.voxel import creation as vox_creation

from augment import random_permute_flat, random_permute_mlp, sorted_permute_mlp
from hd_utils import generate_mlp_from_weights, get_mlp
from siren.dataio import anime_read


class VoxelDataset(Dataset):
    def __init__(
        self, mesh_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        self.mesh_folder = mesh_folder
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))

        self.mesh_files = []
        if object_names is None:
            self.mesh_files = [
                file
                for file in list(os.listdir(mesh_folder))
                if file not in ["train_split.lst", "test_split.lst", "val_split.lst"]
            ]
        else:
            for file in list(os.listdir(mesh_folder)):
                if file.split(".")[0] in blacklist and cfg.filter_bad:
                    continue

                if (
                    ("_" in file and file.split("_")[1] in object_names)
                    or file in object_names
                    or file.split(".")[0] in object_names
                ):
                    self.mesh_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.cfg = cfg
        self.vox_folder = self.mesh_folder + "_vox"
        os.makedirs(self.vox_folder, exist_ok=True)

    def __getitem__(self, index):
        dir = self.mesh_files[index]
        path = join(self.mesh_folder, dir)
        resolution = self.cfg.vox_resolution
        voxel_size = 1.9 / (resolution - 1)
        total_time = self.cfg.unet_config.params.image_size
        if self.cfg.mlp_config.params.move:
            folder_name = os.path.basename(path)
            anime_file_path = os.path.join(path, folder_name + ".anime")
            nf, nv, nt, vert_data, face_data, offset_data = anime_read(anime_file_path)

            def normalize(obj, v_min, v_max):
                vertices = obj.vertices
                vertices -= np.mean(vertices, axis=0, keepdims=True)
                vertices *= 0.95 / (max(abs(v_min), abs(v_max)))
                obj.vertices = vertices
                return obj

            # total_time = min(nf, total_time)
            vert_datas = []
            v_min, v_max = float("inf"), float("-inf")

            frames = np.linspace(0, nf, total_time, dtype=int, endpoint=False)
            if self.cfg.move_sampling == "first":
                frames = np.linspace(
                    0, min(nf, total_time), total_time, dtype=int, endpoint=False
                )

            for t in frames:
                vert_data_copy = vert_data
                if t > 0:
                    vert_data_copy = vert_data + offset_data[t - 1]
                vert_datas.append(vert_data_copy)
                vert = vert_data_copy - np.mean(vert_data_copy, axis=0, keepdims=True)
                v_min = min(v_min, np.amin(vert))
                v_max = max(v_max, np.amax(vert))
            grids = []
            for vert_data in vert_datas:
                obj = trimesh.Trimesh(vert_data, face_data)
                obj = normalize(obj, v_min, v_max)
                voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                    obj, pitch=voxel_size
                )
                voxel_grid.fill()
                grid = voxel_grid.matrix
                padding_amounts = [
                    (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                    for length in grid.shape
                ]
                grid = np.pad(grid, padding_amounts).astype(np.float32)
                grids.append(grid)
            grid = np.stack(grids)
        else:
            mesh: trimesh.Trimesh = trimesh.load(path)
            coords = np.asarray(mesh.vertices)
            coords = coords - np.mean(coords, axis=0, keepdims=True)
            v_max = np.amax(coords)
            v_min = np.amin(coords)
            coords *= 0.95 / (max(abs(v_min), abs(v_max)))
            mesh.vertices = coords
            voxel_grid: trimesh.voxel.VoxelGrid = vox_creation.voxelize(
                mesh, pitch=voxel_size
            )
            voxel_grid.fill()
            grid = voxel_grid.matrix
            padding_amounts = [
                (floor((resolution - length) / 2), ceil((resolution - length) / 2))
                for length in grid.shape
            ]
            grid = np.pad(grid, padding_amounts).astype(np.float32)

        # Convert 0 regions to -1, so that the input is -1 or +1.
        grid[grid == 0] = -1

        grid = torch.tensor(grid).float()

        # Doing some sanity checks for 4D and 3D generations
        if self.cfg.mlp_config.params.move:
            assert (
                grid.shape[0] == total_time
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
                and grid.shape[3] == resolution
            )
            return grid, 0
        else:
            assert (
                grid.shape[0] == resolution
                and grid.shape[1] == resolution
                and grid.shape[2] == resolution
            )

        return grid[None, ...], 0

    def __len__(self):
        return len(self.mesh_files)


class WeightDataset(Dataset):
    def __init__(
        self, mlps_folder, wandb_logger, model_dims, mlp_kwargs, cfg, object_names=None
    ):
        self.mlps_folder = mlps_folder
        self.condition = cfg.transformer_config.params.condition
        files_list = list(os.listdir(mlps_folder))
        blacklist = {}
        if cfg.filter_bad:
            blacklist = set(np.genfromtxt(cfg.filter_bad_path, dtype=str))
        if object_names is None:
            self.mlp_files = [file for file in list(os.listdir(mlps_folder))]
        else:
            self.mlp_files = []
            for file in list(os.listdir(mlps_folder)):
                # Excluding black listed shapes
                if cfg.filter_bad and file.split("_")[1] in blacklist:
                    continue
                # Check if file is in corresponding split (train, test, val)
                # In fact, only train split is important here because we don't use test or val MLP weights
                if ("_" in file and (file.split("_")[1] in object_names or (
                        file.split("_")[1] + "_" + file.split("_")[2]) in object_names)) or (file in object_names):
                    self.mlp_files.append(file)
        self.transform = None
        self.logger = wandb_logger
        self.model_dims = model_dims
        self.mlp_kwargs = mlp_kwargs
        if cfg.augment in ["permute", "permute_same", "sort_permute"]:
            self.example_mlp = get_mlp(mlp_kwargs)
        self.cfg = cfg
        if "first_weight_name" in cfg and cfg.first_weight_name is not None:
            self.first_weights = self.get_weights(
                torch.load(os.path.join(self.mlps_folder, cfg.first_weight_name))
            ).float()
        else:
            self.first_weights = torch.tensor([0])

    def get_weights(self, state_dict):
        weights = []
        shapes = []
        for weight in state_dict:
            shapes.append(np.prod(state_dict[weight].shape))
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights)
        prev_weights = weights.clone()

        # Some augmentation methods are available althougwe don't use them in the main paper
        if self.cfg.augment == "permute":
            weights = random_permute_flat(
                [weights], self.example_mlp, None, random_permute_mlp
            )[0]
        if self.cfg.augment == "sort_permute":
            example_mlp = generate_mlp_from_weights(weights, self.mlp_kwargs)
            weights = random_permute_flat(
                [weights], example_mlp, None, sorted_permute_mlp
            )[0]
        if self.cfg.augment == "permute_same":
            weights = random_permute_flat(
                [weights],
                self.example_mlp,
                int(np.random.random() * self.cfg.augment_amount),
                random_permute_mlp,
            )[0]
        if self.cfg.jitter_augment:
            weights += np.random.uniform(0, 1e-3, size=weights.shape)

        if self.transform:
            weights = self.transform(weights)
        # We also return prev_weights, in case you want to do permutation, we store prev_weights to sanity check later
        return weights, prev_weights

    def __getitem__(self, index):
        file = self.mlp_files[index]
        dir = join(self.mlps_folder, file)
        if os.path.isdir(dir):
            path1 = join(dir, "checkpoints", "model_final.pth")
            path2 = join(dir, "checkpoints", "model_current.pth")
            state_dict = torch.load(path1 if os.path.exists(path1) else path2)
        else:
            state_dict = torch.load(dir, map_location=torch.device("cpu"))

        weights, weights_prev = self.get_weights(state_dict)

        if self.cfg.augment == "inter":
            other_index = np.random.choice(len(self.mlp_files))
            other_dir = join(self.mlps_folder, self.mlp_files[other_index])
            other_state_dict = torch.load(other_dir)
            other_weights, _ = self.get_weights(other_state_dict)
            lerp_alpha = np.random.uniform(
                low=0, high=self.cfg.augment_amount
            )  # Prev: 0.3
            weights = torch.lerp(weights, other_weights, lerp_alpha)

        return weights.float(), weights_prev.float(), weights_prev.float()

    def __len__(self):
        return len(self.mlp_files)
