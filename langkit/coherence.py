from typing import Optional
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

#Text classification model
_coherence_model_path = "madhurjindal/autonlp-Gibberish-Detector-492513457"
_coherence_tokenizer = None
_coherence_pipeline = None


def init(model_path: Optional[str] = None):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    global _coherence_tokenizer, _coherence_pipeline
    if model_path is None:
        model_path = _coherence_model_path
    _coherence_tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _coherence_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_coherence_tokenizer
    )


@register_metric_udf(col_type=String)
def coherence(text: str) -> float:
    if _coherence_pipeline is None or _coherence_tokenizer is None:
        raise ValueError("coherence score must initialize the pipeline first")
    result = _coherence_pipeline(
        text, truncation=True, max_length=_coherence_tokenizer.model_max_length, top_k = None
    )

    for item in result:
        if item['label'] == 'clean':
            clean_score = item['score']
            break  # Exit the loop once the desired label is found
        
    return clean_score


init()