"""Re-save the paper's Eindhoven-trained GRUs as the checkpoints the package ships.

    python scripts/export_gru_checkpoints.py <react-reproducibility>/artifacts/eindhoven/gru_10hz

Keeps only the weights and the settings needed to rebuild and describe the network; drops the
optimizer and scheduler state and every path or device recorded during training, so the files
load with torch.load(weights_only=True).
"""

from __future__ import annotations

import argparse
import os

import torch

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "reactmetric", "baselines",
                   "checkpoints")
KEEP = ("history_length", "forecast_horizon", "frame_dt", "hidden_size", "num_layers",
        "dropout", "min_trajectory_length", "train_ratio", "val_ratio", "sample_stride",
        "split_seed", "batch_size", "learning_rate", "weight_decay", "max_epochs", "patience")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory with gru_position_best.pt and gru_orientation_best.pt")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    for variant in ("position", "orientation"):
        name = f"gru_{variant}_best.pt"
        ckpt = torch.load(os.path.join(args.src, name), map_location="cpu", weights_only=False)
        config = {k: ckpt["config"][k] for k in KEEP if k in ckpt["config"]}
        config.update(training_data="Eindhoven pedestrian tracks (CC-BY-4.0), 10 Hz",
                      epoch=int(ckpt["epoch"]), val_ade=float(ckpt["val_ade"]),
                      val_fde=float(ckpt["val_fde"]), trained_at=ckpt["trained_at"][:10])
        state = {k: v.detach().clone().contiguous() for k, v in ckpt["model_state_dict"].items()}
        out = os.path.join(args.out, name)
        torch.save({"model_state_dict": state, "config": config}, out)
        check = torch.load(out, map_location="cpu", weights_only=True)
        assert all(torch.equal(check["model_state_dict"][k], v) for k, v in state.items())
        print(f"{out}: {os.path.getsize(out) / 1024:.0f} KB, {config}")


if __name__ == "__main__":
    main()
