"""From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
"""
#!/usr/bin/env python3

import logging
import os
import time

import numpy as np
import plyfile
import skimage.measure
import torch


def create_mesh_v2(
    decoder,
    filename,
    N=256,
    max_batch=64**3,
    level=0,
    offset=None,
    scale=None,
    vis_transform=None,
):
    start = time.time()
    ply_filename = filename
    world_to_mc_grid = torch.eye(4)
    world_to_mc_grid[0, 0] = N / 2
    world_to_mc_grid[1, 1] = N / 2
    world_to_mc_grid[2, 2] = N / 2
    world_to_mc_grid[:3, 3] = N / 2
    decoder.eval()

    voxel_origin = [0, 0, 0]
    voxel_size = 1.0 / N
    print("voxel_size", voxel_size)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 0] = overall_index % N
    samples[:, 1] = torch.div(overall_index, N, rounding_mode="floor") % N
    samples[:, 2] = (
        torch.div(
            torch.div(overall_index, N, rounding_mode="floor"), N, rounding_mode="floor"
        )
        % N
    )

    # transform first 3 columns
    # to be the x, y, z coordinate
    mc_grid_to_world = torch.inverse(world_to_mc_grid)
    samples[:, :3] = (
        torch.matmul(
            mc_grid_to_world[:3, :3], samples[:, :3].transpose(1, 0)
        ).transpose(1, 0)
        + mc_grid_to_world[:3, 3]
    )

    num_samples = N**3
    samples.requires_grad = False

    head = 0

    print(
        "samples[:,:3]",
        samples[:, :3].shape,
        torch.min(samples[:, :3]).item(),
        torch.max(samples[:, :3]).item(),
    )

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset).squeeze().detach().cpu()  # .squeeze(1)
        )
        head += max_batch

    sdf_values = samples[:, 3]
    print(
        "sdf_values",
        sdf_values.shape,
        torch.min(sdf_values).item(),
        torch.max(sdf_values).item(),
        torch.mean(sdf_values).item(),
    )
    nz = torch.nonzero(sdf_values < 0)
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    sdf_tensor = sdf_values.detach().cpu().view(N, N, N) / voxel_size
    print(
        "sdf_tensor",
        sdf_tensor.shape,
        torch.min(sdf_tensor).item(),
        torch.max(sdf_tensor).item(),
        torch.mean(sdf_tensor).item(),
    )
    trunc = torch.max(sdf_tensor).item() - 0.1
    v, f = mc.marching_cubes(
        sdf_tensor,
        None,
        transform=torch.inverse(world_to_mc_grid),
        isovalue=level,
        truncation=trunc,
        thresh=10 * trunc,
        output_filename=None if ply_filename is None else (ply_filename + ".ply"),
    )
    return v, f, sdf_tensor
    hist_min = -0.15
    hist_max = 0.15
    num_hist = 20
    inc = (hist_max - hist_min) / float(num_hist)
    hist = np.zeros(num_hist)
    for i in range(num_hist):
        hist[i] = torch.sum(
            torch.logical_and(
                sdf_values > hist_min + i * inc, sdf_values < hist_min + (i + 1) * inc
            )
        ).item()
    splitter = ","
    with open(ply_filename + "_hist.csv", "w") as ofs:
        ofs.write(splitter.join(["range min", "range max", "count"]) + "\n")
        for i in range(num_hist):
            ofs.write(
                splitter.join(
                    [
                        str(x)
                        for x in [hist_min + i * inc, hist_min + (i + 1) * inc, hist[i]]
                    ]
                )
                + "\n"
            )


def create_mesh(
    decoder,
    filename=None,
    N=256,
    max_batch=64**3,
    offset=None,
    scale=None,
    level=0,
    time_val=-1,
):
    start = time.time()
    ply_filename = filename
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-0.5] * 3
    voxel_size = -2 * voxel_origin[0] / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        # print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        if time_val >= 0:
            sample_subset = torch.hstack(
                (
                    sample_subset,
                    torch.ones((sample_subset.shape[0], 1)).cuda() * time_val,
                )
            )
        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset).squeeze().detach().cpu()  # .squeeze(1)
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    end = time.time()
    # print("sampling takes: %f" % (end - start))

    return convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        None if ply_filename is None else ply_filename + ".ply",
        level,
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    level,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    # v, f = mcubes.marching_cubes(mcubes.smooth(pytorch_3d_sdf_tensor.numpy()), 0)
    # mcubes.export_obj(v, f, ply_filename_out.split(".")[-2][1:] + ".obj")
    # return
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    # print(numpy_3d_sdf_tensor.min(), numpy_3d_sdf_tensor.max())
    verts, faces, normals, values = (
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(0),
    )
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print(e)
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    if ply_filename_out is not None:
        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        logging.debug("saving mesh to %s" % (ply_filename_out))
        ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

    return mesh_points, faces, pytorch_3d_sdf_tensor
