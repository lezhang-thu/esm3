import os

os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib

from esm.models.x_esmc import ESMC
from esm.models.x_dataset import SequenceDataset, collate_fn
from esm.sdk.api import ESMProtein

BATCH_SIZE = 2
VAL_BATCH = 4
NUM_EPOCHS = 100
#EVAL_ITER = 4096
EVAL_ITER = 1024
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
        self.standard_normal = torch.distributions.Normal(0, 1)

    def forward(
        self,
        mu,
        logsigma,
        values,
        exact_mask,
        greater_mask,
        less_mask,
    ):
        #logsigma = torch.clamp(logsigma, min=-10, max=10)  # σ ∈ [4.5e-5, 22026]
        #logsigma = torch.clamp(logsigma, min=-7, max=7)  # σ ∈ [9e-4, 1096]
        #sigma = torch.clamp(logsigma.exp(), min=1e-6)
        logsigma = 0.0
        sigma = 1.0
        z = (values - mu) / sigma
        #z = torch.clamp(z, min=-10, max=10)
        z = torch.clamp(z, min=-8, max=8)  # Φ(±8) ≈ 6e-16

        # Exact: -0.5*log(2π) - log(σ) - 0.5*z²
        exact_nll = 0.5 * z**2 + logsigma + 0.5 * torch.log(
            torch.tensor(2 * 3.14159265359))

        # Left-censored: log Φ(z)
        left_log_prob = torch.special.log_ndtr(z)

        # Right-censored: log(1 - Φ(z)) = log Φ(-z)
        right_log_prob = torch.special.log_ndtr(-z)

        # debug
        exact_nll = .5 * (values - mu)**2
        assert torch.all(exact_mask[:, :1])
        nll = exact_nll * exact_mask.float()
        #nll += -left_log_prob * less_mask.float()
        #nll += -right_log_prob * greater_mask.float()
        nll = nll[:, :1]

        return nll.mean()


def evaluate(client, val_loader, criterion):
    client.eval()
    total_nll = 0.0
    count = 0

    for (seq, label) in val_loader:
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        device = protein_tensor.device
        t = []
        for _ in label:
            t.append(torch.from_numpy(_).to(device))
        values, exact_mask, greater_mask, less_mask = t
        with (
                torch.no_grad(),
                torch.autocast(enabled=True,
                               device_type=device.type,
                               dtype=torch.bfloat16)  # type: ignore
                if device.type == "cuda" else contextlib.nullcontext(),
        ):
            mu, logsigma = client.predict(protein_tensor)

            nll = criterion(
                mu,
                logsigma,
                values,
                exact_mask,
                greater_mask,
                less_mask,
            )
        total_nll += nll.item() * mu.size(0)
        count += mu.size(0)

    return total_nll / count


def main(client, train_loader, val_loader):
    lora_params = []
    for name, param in client.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            lora_params.append(param)
    client.mu.requires_grad_(True)
    client.logsigma.requires_grad_(True)

    from itertools import chain
    optimizer = torch.optim.AdamW(
        chain(
            lora_params,
            client.mu.parameters(),
            client.logsigma.parameters(),
        ),
        lr=1e-4,
        weight_decay=0.01,
    )
    criterion = CensoredGaussianNLL()
    for epoch in range(NUM_EPOCHS):
        client.train()
        for idx, (seq, label) in enumerate(train_loader):
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            device = protein_tensor.device
            t = []
            for _ in label:
                t.append(torch.from_numpy(_).to(device))
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
                print('idx: {}'.format(idx))
                print('values[:, :1]:\n{}'.format(
                    values.detach().cpu()[:, :1]))
                print('mu[:, :1]:\n{}'.format(mu.detach().cpu()[:, :1]))
                #print('mu:\n{}'.format(mu.detach().cpu()))
            if (idx + 1) % GRAD_ACC == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (idx + 1) % EVAL_ITER == 0:
                test_nll = evaluate(client, val_loader, criterion)
                print(f"Epoch {epoch}, iter {idx}: Test NLL = {test_nll:.4f}")
                client.train()
    test_nll = evaluate(client, val_loader, criterion)
    print(f"FINAL Test NLL = {test_nll:.4f}")


if __name__ == '__main__':
    import pandas as pd
    prefix = "/home/ubuntu/lezhang.thu/biology-research/hiv-1/hiv-data/antibody-antigen-seq"
    #df = pd.read_csv(os.path.join(prefix, "filtered-assay.csv"))
    df = pd.read_csv(os.path.join(prefix, "filtered-assay-numeric.csv"))
    from sklearn.model_selection import train_test_split

    #train_df, val_df = train_test_split(df, test_size=0.02, random_state=42)
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
    main(client, train_loader, val_loader)
