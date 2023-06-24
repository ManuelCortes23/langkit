from typing import Optional
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

#Text classification model
_irony_model_path = "cardiffnlp/twitter-roberta-base-irony"
_irony_tokenizer = None
_irony_pipeline = None


def init(model_path: Optional[str] = None):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    global _irony_tokenizer, _irony_pipeline
    if model_path is None:
        model_path = _irony_model_path
    _irony_tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _irony_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_irony_tokenizer
    )


@register_metric_udf(col_type=String)
def irony(text: str) -> float:
    if _irony_pipeline is None or _irony_tokenizer is None:
        raise ValueError("irony score must initialize the pipeline first")
    
    result = _irony_pipeline(text, top_k = None)#, truncation = True, max_length = _tokenizer.model_max_length)

    
    irony_score = (
        result[0]["score"] if result[0]["label"] == "irony" else 1 - result[0]["score"]
    )
    return irony_score

init()
