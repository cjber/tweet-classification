import pandas as pd
from pathlib import Path
from src.common.utils import Const
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CSVDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer=AutoTokenizer,
        meta: bool = True,
    ):
        self.data = pd.read_csv(path)
        self.meta = meta

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

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
            "text": text,
        }

        if self.meta:
            item["meta"] = {
                "diff_date": idx["diff_date"] if "diff_date" in idx else None,
                "idx": idx["idx"] if "idx" in idx else None,
            }
        return item
