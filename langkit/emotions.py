from typing import Optional
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

#Text classification model
_model_path = "SamLowe/roberta-base-go_emotions"
_tokenizer = None
_text_classification_pipeline = None


def init(model_path: Optional[str] = None):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    global _tokenizer, _text_classification_pipeline
    if model_path is None:
        model_path = _model_path
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _text_classification_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_tokenizer
    )


@register_metric_udf(col_type=String)
def emotion(text) -> str:
    if _text_classification_pipeline is None or _tokenizer is None:
        raise ValueError("Must initialize emotions udf before evaluation.")

    result = _text_classification_pipeline(
        text, truncation=True, max_length=_tokenizer.model_max_length, top_k = None
    )

    return result[0]['label']

    #Altenatively, we can have each emotion be a feature and return its probability, but that's a lot of features
    emotion_score = (
        result[0]["score"]
    )
    return emotion_score

# #The following would upload the top 10 emotions and give their 'probability' on a scale from [0,10]
# #TODO actualy implement this correctly
# @register_metric_udf(col_type=String)
# def emotion(text) -> float:
#     if _text_classification_pipeline is None or _tokenizer is None:
#         raise ValueError("Must initialize emotions udf before evaluation.")

#     result = _text_classification_pipeline(
#         text, truncation=True, max_length=_tokenizer.model_max_length, top_k = None
#     )

#     emotion_score = (
#         result[0]["score"]
#     )
#     return emotion_score


init()
