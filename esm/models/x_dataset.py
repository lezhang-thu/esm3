import torch
from torch.utils.data import Dataset, DataLoader
import random

import numpy as np
import pandas as pd


class SequenceDataset(Dataset):

    def __init__(self, df):
        import os
        self.data = df
        prefix = "./hiv-data/antibody-antigen-seq"

        t = pd.read_csv(os.path.join(prefix, "antibody-seq.csv"))
        self.antibody_lookup = {
            #row['antibody-id']: row['heavy-seq'].ljust(512, '-') + '|' + row['light-seq'].ljust(256, '-')
            row['antibody-id']: row['heavy-seq'] + '|' + row['light-seq']
            for _, row in t.iterrows()
        }

        t = pd.read_csv(os.path.join(prefix, "virus-seq.csv"))
        self.virus_lookup = {
            #row['virus-id']: row['seq'].ljust(1152, '-')
            row['virus-id']: row['seq']
            for _, row in t.iterrows()
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        antibody, virus, IC50, IC80, ID50 = self.data.iloc[idx]
        subs = [x.strip() for x in antibody.split("+")]
        random.shuffle(subs)
        subs = [self.antibody_lookup[_] for _ in subs]
        #assert len(subs) == 1
        seq = "||".join(subs)

        # first virus, then antibody. as antibodies can be multiple
        seq = "|||".join([self.virus_lookup[virus], seq])
        return seq, (IC50, IC80, ID50)


def collate_fn(batch):
    # batch is a list of tuples: [(seq1, label1), (seq2, label2), ...]
    sequences, labels = zip(*batch)

    n_samples = len(labels)
    values = np.zeros((n_samples, 3))
    exact_mask = np.zeros((n_samples, 3), dtype=bool)
    greater_mask = np.zeros((n_samples, 3), dtype=bool)
    less_mask = np.zeros((n_samples, 3), dtype=bool)

    for i, row in enumerate(labels):
        for j, val in enumerate(row):
            # Check if value is NaN
            if pd.isna(val):
                continue

            # Try to convert to float directly
            try:
                float_val = float(val)
                values[i, j] = float_val
                exact_mask[i, j] = True
                continue
            except (ValueError, TypeError):
                pass

            # Must be a string with inequality
            val_str = str(val).strip()

            # Assert that it contains ">" or "<"
            assert '>' in val_str or '<' in val_str, \
                f"Value '{val}' at position [{i}, {j}] cannot be converted to float and doesn't contain '>' or '<'"

            # Extract numeric value
            if '>' in val_str:
                numeric_str = val_str.replace('>', '').strip()
                float_val = float(numeric_str)
                values[i, j] = float_val
                greater_mask[i, j] = True
            elif '<' in val_str:
                numeric_str = val_str.replace('<', '').strip()
                float_val = float(numeric_str)
                values[i, j] = float_val
                less_mask[i, j] = True
    if True:
        mask = values > 0
        values[mask] = np.log10(values[mask])
    return sequences, (values, exact_mask, greater_mask, less_mask)
