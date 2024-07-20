from transformers import (
    TapasTokenizer,
    TapexTokenizer,
    TapasModel,
    BartForSequenceClassification,
)

text_model_dict = {
    "tapas": {
        "tokenizer": TapasTokenizer.from_pretrained("google/tapas-base"),
        "model": TapasModel.from_pretrained("google/tapas-base"),
        "embedding_dim": 768,
    },
    "tapex": {
        "tokenizer": TapexTokenizer.from_pretrained(
            "microsoft/tapex-base"
        ),
        "model": BartForSequenceClassification.from_pretrained(
            "microsoft/tapex-base-finetuned-tabfact"
        ),
        "embedding_dim": 768,
    },
}