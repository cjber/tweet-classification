import re
import string

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=[
        "email",
        "percent",
        "money",
        "phone",
        "time",
        "date",
        "number",
    ],
    dicts=[emoticons],
)


class Const:
    MAX_TOKEN_LEN: int = 128
    MODEL_NAME: str = "cardiffnlp/twitter-roberta-base"
    LABELS: dict[str, int] = {
        "B-PLACE": 0,
        "I-PLACE": 1,
        "L-PLACE": 2,
        "U-PLACE": 3,
        "O": 4,
    }


def preprocess(text, processor=text_processor):
    new_text = []
    for t in text.split(" "):
        t = "[USER]" if t.startswith("@") and len(t) > 1 else t
        t = "[HTTP]" if t.startswith("http") else t
        t = f"<{' '.join(re.split('(?=[A-Z])', t[1:]))}>" if t.startswith("#") else t
        t = "" if t == "[HTTP]" else t
        new_text.append(t)
    text = " ".join(new_text).encode("ascii", errors="ignore").decode()

    return processor.pre_process_doc(text)


class Label:
    def __init__(self, name: str):
        """
        Class used to create labels based on task.

        Parameters
        ----------
        name : str
            Name of task, either GER or REL.
        """
        self.name = name
        assert self.name in ["GER", "REL"], "Type must be either GER or REL"

        if self.name == "GER":
            self.labels: dict[str, int] = {
                "B-PLACE": 0,
                "I-PLACE": 1,
                "L-PLACE": 2,
                "U-PLACE": 3,
                "O": 4,
            }
        elif self.name == "REL":
            self.labels: dict[str, int] = {"NONE": 0, "NTPP": 1, "DC": 2}

        self.idx: dict[int, str] = {v: k for k, v in self.labels.items()}
        self.count: int = len(self.labels)


def combine_subwords(tokens: list[str], tags: list[int]) -> tuple[list[str], list[str]]:
    """
    Combines subwords and their tags into normal words with special chars removed.

    WARNING: This removed all punctuation!

    Parameters
    ----------
    tokens : list[str]
        Subword tokens.
    tags : list[int]
        Token tags of same length.

    Returns
    -------
    tuple[list[str], list[str]]:
        Combined tokens and tags.

    Example
    -------

    >>> tokens = ['ĠVery', 'long', 'word', 'Ġfor', 'Ġdoct', 'est', 'Ġ.']
    >>> tags = [1, -100, -100, 0, 1, -100, 0]
    >>> tokens, tags = combine_subwords(tokens, tags)

    >>> tokens
    ['Verylongword', 'for', 'doctest', '.']
    >>> len(tags) == len(tokens)
    True
    """
    idx = [
        idx for idx, token in enumerate(tokens) if token not in ["<pad>", "<s>", "</s>"]
    ]
    tokens = [tokens[i] for i in idx]
    tags = [tags[i] for i in idx]

    for idx, _ in enumerate(tokens):
        idx += 1
        if (
            tokens[-idx + 1][0] != "\u0120"
            and tokens[-idx + 1] not in string.punctuation
        ):
            tokens[-idx] = tokens[-idx] + tokens[-idx + 1]
    idx = [idx for idx, token in enumerate(tokens) if token[0] == "\u0120"]

    tokens = [tokens[i][1:] for i in idx]
    tags_str: list[str] = [Label("GER").idx[tags[i]] for i in idx]
    return tokens, tags_str


def combine_biluo(tokens: list[str], tags: list[str]) -> tuple[list[str], list[str]]:
    """
    Combines multi-token BILUO tags into single entities.

    Parameters
    ----------
    tokens : list[str]
        Input tokenized string.
    tags : list[str]
        Tags corresponding with each token with BILUO format.

    Returns
    -------
    tuple[list[str], list[str]]:
        Tokens and tags with BILUO removed.

    Example:

    >>> tokens = ['New', 'York', 'is', 'big', '.']
    >>> tags = ['B-PLACE', 'L-PLACE', 'O', 'O', 'O']
    >>> tokens, tags = combine_biluo(tokens, tags)

    >>> tokens
    ['New York', 'is', 'big', '.']
    >>> tags
    ['PLACE', 'O', 'O', 'O']
    """
    tokens_biluo = tokens.copy()
    tags_biluo = tags.copy()

    for idx, tag in enumerate(tags_biluo):
        if idx + 1 < len(tags_biluo) and tag[0] == "B":
            i = 1
            while tags_biluo[idx + i][0] not in ["B", "O", "U"]:
                tokens_biluo[idx] = tokens_biluo[idx] + " " + tokens_biluo[idx + i]
                i += 1
                if idx + i == len(tokens_biluo):
                    break
    zipped = [
        (token, tag)
        for (token, tag) in zip(tokens_biluo, tags_biluo)
        if tag[0] not in ["I", "L"]
    ]
    tokens_biluo, tags_biluo = zip(*zipped)
    tags_biluo = [tag[2:] if tag != "O" else tag for tag in tags_biluo]
    return list(tokens_biluo), tags_biluo
