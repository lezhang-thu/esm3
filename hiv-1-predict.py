import os

os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import numpy as np

from esm.models.x_esmc import ESMC
from esm.models.predict_dataset import SequenceDataset
from esm.sdk.api import ESMProtein

VAL_BATCH = 4


def main(client, val_loader):
    client.eval()
    muS = list()
    for seq in val_loader:
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        device = protein_tensor.device
        with (
                torch.no_grad(),
                torch.autocast(enabled=True,
                               device_type=device.type,
                               dtype=torch.bfloat16)  # type: ignore
                if device.type == "cuda" else contextlib.nullcontext(),
        ):
            mu, logsigma = client.predict(protein_tensor)
            muS.append(mu.cpu().float().numpy())
    # NO ID50
    return 10**(np.concatenate(muS, axis=0)[:, :2])


def load_checkpoint(client, ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    print(ckpt.keys())
    with torch.no_grad():
        for name, param in client.named_parameters():
            if 'lora' in name and name in ckpt["lora_params"]:
                param.copy_(ckpt["lora_params"][name].to(param.device))
    client.mu.load_state_dict(ckpt["mu"])
    print(f"Loaded checkpoint from {ckpt_path}")
    return client


if __name__ == '__main__':
    import pandas as pd
    prefix = "."
    val_df = pd.read_csv(os.path.join(prefix, "predict.csv"))
    val_loader = torch.utils.data.DataLoader(
        SequenceDataset(val_df),
        batch_size=VAL_BATCH,
        shuffle=False,
    )
    client = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"
    load_checkpoint(client, "hiv-1.pt")
    muS = main(client, val_loader)
    # Add IC50 and IC80 columns
    val_df["IC50"] = muS[:, 0]
    val_df["IC80"] = muS[:, 1]

    # Save to a new CSV
    out_path = os.path.join(prefix, "predict-ret.csv")
    val_df.to_csv(out_path, index=False)
    print(f"Saved predictions with IC50 and IC80 to {out_path}")
