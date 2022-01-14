import imgkit
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

from src.common.utils import Const
from src.pl_module.classifier_model import FloodModel

test_results: pd.DataFrame = pd.read_csv("data/out/test_results.csv")
full_labelled = pd.read_csv("data/out/full_labelled.csv")

true_negative = test_results[
    (test_results["pred"] == 0)
    & (test_results["truth"] == 0)
    & (test_results["rule"] == 1)
]

true_positive = test_results[
    (test_results["pred"] == 1)
    & (test_results["truth"] == 1)
    & (test_results["rule"] == 0)
]

rule_true_pos = test_results[
    (test_results["pred"] == 0)
    & (test_results["rule"] == 1)
    & (test_results["truth"] == 1)
]

rule_true_neg = test_results[
    (test_results["pred"] == 1)
    & (test_results["rule"] == 0)
    & (test_results["truth"] == 0)
]

model = FloodModel.load_from_checkpoint(
    "ckpts/default/0/checkpoints/checkpoint.ckpt"
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(Const.MODEL_NAME)

cls_explainer = SequenceClassificationExplainer(model.model, tokenizer)

word_attributions = cls_explainer(true_negative.iloc[1]["text"], class_name="NOT_FLOOD")
cls_explainer.visualize("paper/figures/XX_transformer_viz_tn.html")
imgkit.from_url(
    "paper/figures/XX_transformer_viz_tn.html",
    output_path="paper/figures/XX_transformer_viz_tn.png",
)

word_attributions = cls_explainer(true_positive.iloc[0]["text"], class_name="FLOOD")
cls_explainer.visualize("paper/figures/XX_transformer_viz_tp.html")
imgkit.from_url(
    "paper/figures/XX_transformer_viz_tp.html",
    output_path="paper/figures/XX_transformer_viz_tp.png",
)

word_attributions = cls_explainer(rule_true_pos.iloc[0]["text"], class_name="FLOOD")
cls_explainer.visualize("paper/figures/XX_transformer_viz_rtp.html")
imgkit.from_url(
    "paper/figures/XX_transformer_viz_rtp.html",
    output_path="paper/figures/XX_transformer_viz_rtp.png",
)

word_attributions = cls_explainer(rule_true_neg.iloc[0]["text"], class_name="NOT_FLOOD")
cls_explainer.visualize("paper/figures/XX_transformer_viz_rtn.html")
imgkit.from_url(
    "paper/figures/XX_transformer_viz_rtn.html",
    output_path="paper/figures/XX_transformer_viz_rtn.png",
)

full_floods = full_labelled[full_labelled["label"] == "FLOOD"]["text"].tolist()


word_attributions = []
for item in tqdm(full_floods):
    word_attributions.extend(cls_explainer(item, class_name="FLOOD"))


flood_attributions = pd.DataFrame(word_attributions).rename(
    columns={0: "word", 1: "attribution"}
)
flood_attributions = flood_attributions[~flood_attributions["word"].str.contains("##")]
flood_pos = (
    flood_attributions.groupby("word")
    .mean()
    .sort_values(by="attribution", ascending=False)
    .head(10)
)
flood_pos.to_csv("data/out/flood_pos.csv")

flood_neg = (
    flood_attributions.groupby("word")
    .mean()
    .sort_values(by="attribution", ascending=True)
    .head(10)
)
flood_neg.to_csv("data/out/flood_neg.csv")
