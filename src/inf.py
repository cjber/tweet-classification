from pathlib import Path

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.pl_data.csv_dataset import CSVDataset
from src.pl_module.classifier_model import FloodModel

KEYWORDS = [line.strip() for line in open("data/floods/flood_lexicon.txt")]


def load_model(model_path: str):
    model = FloodModel.load_from_checkpoint(model_path)
    model.eval()
    model.to("cuda")
    model.freeze()
    return model


def load_data(path: Path, batch_size: int):
    data = CSVDataset(path=path)
    return DataLoader(data, shuffle=True, batch_size=batch_size)


def test_model(data, model, keywords):
    results = []
    for item in tqdm(data):
        output = model(
            item["input_ids"].to("cuda"),
            item["attention_mask"].to("cuda"),
            item["text"],
        )
        text = item["text"][0]
        label = torch.argmax(output["logits"].squeeze()).tolist()
        truth = item["labels"].numpy().item()
        rule = 1 if any(k in text.lower() for k in keywords) else 0
        results.append({"text": text, "pred": label, "truth": truth, "rule": rule})
    results = pd.DataFrame(results)
    return results


if __name__ == "__main__":
    model = load_model(
        model_path=str(Path("./ckpts/default/0/checkpoints/checkpoint.ckpt"))
    )
    test_data = load_data(path=Path("data/train/val_data.csv"), batch_size=1)

    test_results = test_model(test_data, model, keywords=KEYWORDS)
    test_results.to_csv("data/out/test_results.csv", index=False)
