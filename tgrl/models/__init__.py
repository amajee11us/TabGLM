from transformers import (
    TapasTokenizer,
    TapasModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

text_model_dict = {
    "tapas": {
        "tokenizer": TapasTokenizer.from_pretrained("google/tapas-base"),
        "model": TapasModel.from_pretrained("google/tapas-base"),
        "embedding_dim": 768,
    },
    "tapex": {
        "tokenizer": AutoTokenizer.from_pretrained(
            "microsoft/tapex-large-finetuned-tabfact"
        ),
        "model": AutoModelForSequenceClassification.from_pretrained(
            "microsoft/tapex-large-finetuned-tabfact"
        ),
        "embedding_dim": 1024,
    },
}
