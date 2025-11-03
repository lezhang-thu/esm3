import os

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib

from esm.models.x_esmc import ESMC
from esm.models.x_dataset import SequenceDataset, collate_fn
from esm.sdk.api import ESMProtein

BATCH_SIZE = 4
VAL_BATCH = 4
NUM_EPOCHS = 100
EVAL_ITER = 4096
GRAD_ACC = 16

#EPS = 1e-12
#
#
#class CensoredGaussianNLL(nn.Module):
#
#    def __init__(self):
#        super().__init__()
#
#    def log_cdf(self, normal: torch.distributions.Normal, x):
#        # numerically stable log CDF using torch.clamp
#        cdf = torch.clamp(normal.cdf(x), min=EPS, max=1.0 - EPS)
#        return torch.log(cdf)
#
#    def log_sf(self, normal: torch.distributions.Normal, x):
#        # log survival = log(1 - CDF)
#        sf = torch.clamp(1.0 - normal.cdf(x), min=EPS, max=1.0 - EPS)
#        return torch.log(sf)
#
#    def forward(
#        self,
#        mu,
#        logsigma,
#        values,
#        exact_mask,
#        greater_mask,
#        less_mask,
#    ):
#        sigma = torch.clamp(logsigma.exp(), min=1e-6)  # ensure positivity
#        normal = torch.distributions.Normal(mu, sigma)
#        nll = -normal.log_prob(values) * exact_mask.float()
#        nll += -self.log_cdf(normal, values) * less_mask.float()
#        nll += -self.log_sf(normal, values) * greater_mask.float()
#        return nll.mean()


class CensoredGaussianNLL(nn.Module):

    def __init__(self):
        super().__init__()
        #self.standard_normal = torch.distributions.Normal(0, 1)
        self.t = 0.5 * torch.log(torch.tensor(2 * torch.pi, device='cuda'))

    def forward(
        self,
        mu,
        logsigma,
        values,
        exact_mask,
        greater_mask,
        less_mask,
    ):
        total_loss = 0.0

        # Exact observations
        if exact_mask.any():
            total_loss += .5 * torch.sum(
                (values[exact_mask] - mu[exact_mask])**2)

        # Left-censored (upper bound)
        if less_mask.any():
            total_loss += .5 * torch.sum(
                torch.relu(mu[less_mask] - values[less_mask])**2)

        # Right-censored (lower bound)
        if greater_mask.any():
            total_loss += .5 * torch.sum(
                torch.relu(values[greater_mask] - mu[greater_mask])**2)

        # Normalize by total number of positions
        denom = mu.shape[0] * mu.shape[1]
        return total_loss / denom

    def separate_nll(self, mu, logsigma, values, exact_mask, greater_mask,
                     less_mask):
        # --- Exact NLL ---
        exact_loss = torch.tensor(0.0, device=mu.device)
        if exact_mask.any():
            exact_loss = .5 * torch.sum(
                (values[exact_mask] - mu[exact_mask])**2)

        # --- Left-censored NLL ---
        left_loss = torch.tensor(0.0, device=mu.device)
        if less_mask.any():
            left_loss = .5 * torch.sum(
                torch.relu(mu[less_mask] - values[less_mask])**2)

        # --- Right-censored NLL ---
        right_loss = torch.tensor(0.0, device=mu.device)
        if greater_mask.any():
            right_loss = .5 * torch.sum(
                torch.relu(values[greater_mask] - mu[greater_mask])**2)

        return exact_loss, left_loss, right_loss


def evaluate(client, val_loader, criterion):
    client.eval()
    # Index 0: exact, 1: left-censored, 2: right-censored
    total_losses = [0.0, 0.0, 0.0]
    total_counts = [0, 0, 0]
    for (seq, label) in val_loader:
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        device = protein_tensor.device
        t = [torch.from_numpy(_).to(device) for _ in label]
        values, exact_mask, greater_mask, less_mask = t
        with (
                torch.no_grad(),
                torch.autocast(enabled=True,
                               device_type=device.type,
                               dtype=torch.bfloat16)  # type: ignore
                if device.type == "cuda" else contextlib.nullcontext(),
        ):
            mu, logsigma = client.predict(protein_tensor)
            losses = criterion.separate_nll(
                mu,
                logsigma,
                values,
                exact_mask,
                greater_mask,
                less_mask,
            )

        masks = [exact_mask, less_mask, greater_mask]
        for i in range(3):
            total_losses[i] += losses[i].item()
            total_counts[i] += masks[i].sum().item()
    # Compute averages, guarding against divide-by-zero
    avg_losses = [
        total_losses[i] / total_counts[i] if total_counts[i] > 0 else 0.0
        for i in range(3)
    ]
    # Global weighted average
    total_loss_sum = sum(total_losses)
    total_count_sum = sum(total_counts)
    global_avg_nll = total_loss_sum / total_count_sum if total_count_sum > 0 else 0.0

    #return tuple(avg_losses)  # (avg_exact, avg_left, avg_right)
    # Return all four values
    return (*avg_losses, global_avg_nll)


def main(client, train_loader, val_loader):
    lora_params = []
    for name, param in client.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            lora_params.append(param)
    client.mu.requires_grad_(True)
    #client.logsigma.requires_grad_(True)

    from itertools import chain
    optimizer = torch.optim.AdamW(
        chain(
            lora_params,
            client.mu.parameters(),
            #client.logsigma.parameters(),
        ),
        lr=1e-4,
        weight_decay=0.01,
    )
    criterion = CensoredGaussianNLL()

    best_nll = float("inf")  # track best validation performance
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    exact_nll, left_nll, right_nll, global_nll = evaluate(
        client,
        val_loader,
        criterion,
    )
    print(f"  Exact NLL = {exact_nll:.4f}")
    print(f"  Left-censored NLL = {left_nll:.4f}")
    print(f"  Right-censored NLL = {right_nll:.4f}")
    print(f"  --> Global Avg NLL = {global_nll:.4f}")

    for epoch in range(NUM_EPOCHS):
        client.train()
        for idx, (seq, label) in enumerate(train_loader):
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            device = protein_tensor.device
            t = [torch.from_numpy(_).to(device) for _ in label]
            values, exact_mask, greater_mask, less_mask = t
            with (
                    torch.autocast(enabled=True,
                                   device_type=device.type,
                                   dtype=torch.bfloat16)  # type: ignore
                    if device.type == "cuda" else contextlib.nullcontext(), ):
                mu, logsigma = client.predict(protein_tensor)
                loss = criterion(
                    mu,
                    logsigma,
                    values,
                    exact_mask,
                    greater_mask,
                    less_mask,
                )
            (loss / GRAD_ACC).backward()
            if (idx + 1) % 128 == 0:
                print('#' * 20)
                print('idx: {}'.format(idx))
                print('values:\n{}'.format(values.detach().cpu()))
                print('mu:\n{}'.format(mu.detach().cpu()))
                #print('values[:, :1]:\n{}'.format(
                #    values.detach().cpu()[:, :1]))
                #print('mu[:, :1]:\n{}'.format(mu.detach().cpu()[:, :1]))
                #print('mu:\n{}'.format(mu.detach().cpu()))
            if (idx + 1) % GRAD_ACC == 0 or (idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            if (idx + 1) % EVAL_ITER == 0:
                exact_nll, left_nll, right_nll, global_nll = evaluate(
                    client,
                    val_loader,
                    criterion,
                )
                print(f"Epoch {epoch}, iter {idx}:")
                print(f"  Exact NLL = {exact_nll:.4f}")
                print(f"  Left-censored NLL = {left_nll:.4f}")
                print(f"  Right-censored NLL = {right_nll:.4f}")
                print(f"  --> Global Avg NLL = {global_nll:.4f}")

                # Save if global average improves
                if global_nll < best_nll:
                    best_nll = global_nll
                    ckpt_path = os.path.join(save_dir, "best-hiv-1.pt")
                    torch.save(
                        {
                            "lora_params": {
                                n: p.cpu()
                                for n, p in client.named_parameters()
                                if 'lora' in n
                            },
                            "mu": client.mu.state_dict(),
                            #"logsigma": client.logsigma.state_dict(),
                        },
                        ckpt_path)
                    print(f"Saved improved model to {ckpt_path}")

                client.train()
    if False:
        ckpt_path = os.path.join(save_dir, "best-hiv-1.pt")
        torch.save(
            {
                "lora_params": {
                    n: p.cpu()
                    for n, p in client.named_parameters() if 'lora' in n
                },
                "mu": client.mu.state_dict(),
                #"logsigma": client.logsigma.state_dict(),
            },
            ckpt_path)
        print(f"Saved improved model to {ckpt_path}")
    #test_nll = evaluate(client, val_loader, criterion)
    #print(f"FINAL Test NLL = {test_nll:.4f}")


def load_checkpoint(client, ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    with torch.no_grad():
        for name, param in client.named_parameters():
            if 'lora' in name and name in ckpt["lora_params"]:
                param.copy_(ckpt["lora_params"][name].to(param.device))
    client.mu.load_state_dict(ckpt["mu"])
    #client.logsigma.load_state_dict(ckpt["logsigma"])
    print(f"Loaded checkpoint from {ckpt_path}")
    return client


if __name__ == '__main__':
    import pandas as pd
    prefix = "./hiv-data/antibody-antigen-seq"
    df = pd.read_csv(os.path.join(prefix, "filtered-assay.csv"))
    #df = pd.read_csv(os.path.join(prefix, "single-assay_noID50.csv"))
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(df, test_size=0.02, random_state=42)
    train_loader = torch.utils.data.DataLoader(
        SequenceDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        SequenceDataset(val_df),
        batch_size=VAL_BATCH,
        collate_fn=collate_fn,
    )
    client = ESMC.from_pretrained("esmc_600m").to("cuda")  # or "cpu"
    if False:
        load_checkpoint(client, "star-hiv-1.pt")
    main(client, train_loader, val_loader)
