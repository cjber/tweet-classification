from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.common.utils import Const


class CSVDataset(Dataset):
    def __init__(self, path: Path, tokenizer=AutoTokenizer):
        self.data = pd.read_csv(path)

        self.tokenizer = tokenizer.from_pretrained(
            Const.MODEL_NAME, add_prefix_space=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        idx = self.data.iloc[index]
        text: str = idx["data"]

        if "label" in idx:
            label = 1 if idx["label"] == "FLOOD" else 0
        else:
            label = -100

        encoding = self.tokenizer(
            text,
            return_attention_mask=True,
            max_length=Const.MAX_TOKEN_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
            "text": text,
        }
