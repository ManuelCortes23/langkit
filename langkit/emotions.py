from typing import Optional
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

#Text classification model
_model_path = "SamLowe/roberta-base-go_emotions"
_tokenizer = None
_text_classification_pipeline = None

#Text classification model
_emotion_model_path = "j-hartmann/emotion-english-distilroberta-base"
_emotion_tokenizer = None
_emotion_pipeline = None

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

    global _emotion_tokenizer, _emotion_pipeline
    #if model_path is None:
    model_path = _emotion_model_path
    _emotion_tokenizer = AutoTokenizer.from_pretrained(_emotion_model_path)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(_emotion_model_path)
    _emotion_pipeline = TextClassificationPipeline(
        model=emotion_model, tokenizer=_emotion_tokenizer
    )


@register_metric_udf(col_type=String)
def emotion(text) -> str:
    if _text_classification_pipeline is None or _tokenizer is None:
        raise ValueError("Must initialize emotions udf before evaluation.")

    result = _text_classification_pipeline(
        text, truncation=True, max_length=_tokenizer.model_max_length, top_k = None
    )

    return result[0]['label']



@register_metric_udf(col_type=String)
def emotion_2(text: str) -> str:
    if _emotion_pipeline is None or _emotion_tokenizer is None:
        raise ValueError("emotion score must initialize the pipeline first")
    result = _emotion_pipeline(
        text, truncation=True, max_length=_emotion_tokenizer.model_max_length, top_k = None
    )

    return result[0]['label']


init()


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

