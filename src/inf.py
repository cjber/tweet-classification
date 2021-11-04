import pandas as pd
import torch
from pathlib import Path
from src.datasets.csv_dataset import CSVDataset
from src.modules.classifier_model import FloodModel
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

KEYWORDS = [
    "flood",
    "floods",
    "flooding",
    "flooded",
    "rain",
    "raining",
    "rains",
    "rained",
    "storm",
    "stormy",
    "thunder",
    "lightning",
]


def load_model(model_path: str):
    model = FloodModel.load_from_checkpoint(model_path)
    model.eval()
    model.to("cuda")
    model.freeze()
    return model


def load_data(path: Path, batch_size: int):
    data = CSVDataset(path=path, meta=False)
    return DataLoader(data, shuffle=True, batch_size=batch_size)


def test_model(data, model, keywords):
    results = []
    for item in tqdm(data):
        output = model(
            item["input_ids"].to("cuda"),
            item["attention_mask"].to("cuda"),
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
        model_path=str(Path("./default/0/checkpoints/epoch=1-step=213.ckpt"))
    )
    test_data = load_data(path=Path("data/train/test.csv"), batch_size=1)

    test_results = test_model(test_data, model, keywords=KEYWORDS)
    test_results.to_csv("data/out/test_results.csv", index=False)

    full_data = load_data(path=Path("./data/out/full_data_clean.csv"), batch_size=16)

    labels_list = []
    text_list = []
    diff_date_list = []
    idx_list = []
    for item in tqdm(full_data):
        outputs = model(item["input_ids"].to("cuda"), item["attention_mask"].to("cuda"))
        labels = outputs["logits"].argmax(dim=1).tolist()
        labels = ["FLOOD" if label == 1 else "NOT_FLOOD" for label in labels]
        labels_list.extend(labels)
        text_list.extend(item["meta"]["text"])
        diff_date_list.extend(item["meta"]["diff_date"])
        idx_list.extend(item["meta"]["idx"].numpy().tolist())

    results = pd.DataFrame(
        {
            "text": text_list,
            "label": labels_list,
            "diff_date": diff_date_list,
            "idx": idx_list,
        }
    )

    results.to_csv("data/out/full_labelled.csv", index=False)
