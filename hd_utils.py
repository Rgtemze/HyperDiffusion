from math import ceil

import numpy as np
import pyrender
import torch
import trimesh

from mlp_models import MLP, MLP3D
from Pointnet_Pointnet2_pytorch.log.classification.pointnet2_ssg_wo_normals import \
    pointnet2_cls_ssg
from torchmetrics_fid import FrechetInceptionDistance

# Using edited 2D-FID code of torch_metrics
fid = FrechetInceptionDistance(reset_real_features=True)


def calculate_fid_3d(
    sample_pcs,
    ref_pcs,
    wandb_logger,
    path="Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth",
):
    batch_size = 10
    point_net = pointnet2_cls_ssg.get_model(40, normal_channel=False)
    checkpoint = torch.load(path)
    point_net.load_state_dict(checkpoint["model_state_dict"])
    point_net.eval().to(sample_pcs.device)
    count = len(sample_pcs)
    for i in range(ceil(count / batch_size)):
        if i * batch_size >= count:
            break
        print(
            ref_pcs[i * batch_size : (i + 1) * batch_size].shape,
            i * batch_size,
            (i + 1) * batch_size,
        )
        real_features = point_net(
            ref_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fake_features = point_net(
            sample_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fid.update(real_features, real=True, features=real_features)
        fid.update(fake_features, real=False, features=fake_features)

    x = fid.compute()
    fid.reset()
    print("x fid_value", x)
    return x


class Config:
    config = None

    @staticmethod
    def get(param):
        return Config.config[param] if param in Config.config else None


def state_dict_to_weights(state_dict):
    weights = []
    for weight in state_dict:
        weights.append(state_dict[weight].flatten())
    weights = torch.hstack(weights)
    return weights


def get_mlp(mlp_kwargs):
    if "model_type" in mlp_kwargs:
        if mlp_kwargs.model_type == "mlp_3d":
            mlp = MLP3D(**mlp_kwargs)
    else:
        mlp = MLP(**mlp_kwargs)
    return mlp


def generate_mlp_from_weights(weights, mlp_kwargs):
    mlp = get_mlp(mlp_kwargs)
    state_dict = mlp.state_dict()
    weight_names = list(state_dict.keys())
    for layer in weight_names:
        val = state_dict[layer]
        num_params = np.product(list(val.shape))
        w = weights[:num_params]
        w = w.view(*val.shape)
        state_dict[layer] = w
        weights = weights[num_params:]
    assert len(weights) == 0, f"len(weights) = {len(weights)}"
    mlp.load_state_dict(state_dict)
    return mlp


def render_meshes(meshes):
    out_imgs = []
    for mesh in meshes:
        img, _ = render_mesh(mesh)
        out_imgs.append(img)
    return out_imgs


def render_mesh(obj):
    if isinstance(obj, trimesh.Trimesh):
        # Handle mesh rendering
        mesh = pyrender.Mesh.from_trimesh(
            obj,
            material=pyrender.MetallicRoughnessMaterial(
                alphaMode="BLEND",
                baseColorFactor=[1, 0.3, 0.3, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8,
            ),
        )
    else:
        # Handle point cloud rendering, (converting it into a mesh instance)
        pts = obj
        sm = trimesh.creation.uv_sphere(radius=0.01)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts
        mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    eye = np.array([2, 1.4, -2])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    camera_pose = look_at(eye, target, up)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(800, 800)
    color, depth = r.render(scene)
    r.delete()
    return color, depth


# Calculate look-at matrix for rendering
def look_at(eye, target, up):
    forward = eye - target
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    camera_pose = np.eye(4)
    camera_pose[:-1, 0] = right
    camera_pose[:-1, 1] = up
    camera_pose[:-1, 2] = forward
    camera_pose[:-1, 3] = eye
    return camera_pose
