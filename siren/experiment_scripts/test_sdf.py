"""Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
"""

# Enable import from parent package
import os
import sys
from pathlib import Path

from mlp_models import MLP3D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from siren import sdf_meshing, utils


class SDFDecoder(torch.nn.Module):
    def __init__(self, model_type, checkpoint_path, mode, cfg):
        super().__init__()
        # Define the model.
        if model_type == "mlp_3d":
            if "mlp_config" in cfg:
                self.model = MLP3D(**cfg.mlp_config)
            else:
                self.model = MLP3D(**cfg)

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {"coords": coords}
        return self.model(model_in)["model_out"]


def main():
    import configargparse

    p = configargparse.ArgumentParser()
    p.add(
        "-c",
        "--config_filepath",
        required=False,
        is_config_file=True,
        help="Path to config file.",
    )

    p.add_argument(
        "--logging_root", type=str, default="./logs", help="root for logging"
    )
    p.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of subdirectory in logging_root where summaries and checkpoints will be saved.",
    )

    # General training options
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument(
        "--checkpoint_path", default=None, help="Checkpoint to trained model."
    )

    p.add_argument(
        "--model_type",
        type=str,
        default="sine",
        help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)',
    )
    p.add_argument(
        "--mode", type=str, default="mlp", help='Options are "mlp" or "nerf"'
    )
    p.add_argument("--resolution", type=int, default=1600)

    opt = p.parse_args()

    sdf_decoder = SDFDecoder(opt.model_type, opt.checkpoint_path, opt.mode)
    name = Path(opt.checkpoint_path).stem
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    sdf_meshing.create_mesh(
        sdf_decoder, os.path.join(root_path, name), N=opt.resolution
    )


if __name__ == "__main__":
    main()
