import torch
from torch.utils.data import Dataset

import pandas as pd


class SequenceDataset(Dataset):

    def __init__(self, df):
        import os
        self.data = df
        prefix = "./hiv-data/antibody-antigen-seq"

        t = pd.read_csv(os.path.join(prefix, "antibody-seq.csv"))
        self.antibody_lookup = {
            #row['antibody-id']: row['heavy-seq'].ljust(512, '-') + '|' + row['light-seq'].ljust(256, '-')
            row['antibody-id']:
            row['heavy-seq'] + '|' + row['light-seq']
            for _, row in t.iterrows()
        }

        t = pd.read_csv(os.path.join(prefix, "virus-seq.csv"))
        self.virus_lookup = {
            #row['virus-id']: row['seq'].ljust(1152, '-')
            row['virus-id']:
            row['seq']
            for _, row in t.iterrows()
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        antibody, virus = self.data.iloc[idx][:2]
        subs = [x.strip() for x in antibody.split("+")]
        subs = [self.antibody_lookup[_] for _ in subs]
        #assert len(subs) == 1
        seq = "||".join(subs)

        # first virus, then antibody. as antibodies can be multiple
        seq = "|||".join([self.virus_lookup[virus], seq])
        return seq
