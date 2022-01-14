import pytorch_lightning as pl
from src.common.utils import Const
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1
from transformers import AutoModelForSequenceClassification
from typing import Any, Union


class FloodModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = Const.MODEL_NAME
        self.optim = AdamW
        self.scheduler = ReduceLROnPlateau

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            return_dict=True,
            output_attentions=True,
            num_labels=2,
            id2label={0: "NOT_FLOOD", 1: "FLOOD"},
            label2id={"NOT_FLOOD": 0, "FLOOD": 1},
        )

        self.train_f1 = F1(num_classes=2)
        self.valid_f1 = F1(num_classes=2)

    def forward(self, input_ids, attention_mask, text, labels=None) -> dict:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def step(self, batch: Any) -> dict:
        outputs = self(**batch)
        softmax = nn.Softmax(dim=1)
        probs = softmax(outputs["logits"])
        return {"loss": outputs["loss"], "probs": probs}

    def training_step(self, batch: Any, _) -> dict:
        step_out = self.step(batch)
        train_f1 = self.train_f1(step_out["probs"], batch["labels"])

        self.log("train_loss", step_out["loss"])
        self.log("train_f1", train_f1)
        return {"loss": step_out["loss"]}

    def validation_step(self, batch: Any, _) -> dict:
        step_out = self.step(batch)
        val_f1 = self.valid_f1(step_out["probs"], batch["labels"])

        self.log("val_loss", step_out["loss"])
        self.log("val_f1", val_f1, prog_bar=True)
        return {"val_loss": step_out["loss"]}

    def configure_optimizers(self) -> Union[Optimizer, dict]:
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if all(nd not in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        opt = self.optim(lr=4e-5, params=optimizer_grouped_parameters)
        scheduler = self.scheduler(optimizer=opt, patience=2, verbose=True)
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
