from typing import Optional
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

_lexicon = "vader_lexicon"
_sentiment_analyzer = None
_nltk_downloaded = False

#Text classification model
_sentiment_model_path = "distilbert-base-uncased-finetuned-sst-2-english"
_sentiment_tokenizer = None
_sentiment_pipeline = None

def init(lexicon: Optional[str] = None, model_path: Optional[str] = None):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    global _sentiment_analyzer, _nltk_downloaded
    if lexicon is None:
        lexicon = _lexicon
    if not _nltk_downloaded:
        nltk.download(lexicon)
        _nltk_downloaded = True
    _sentiment_analyzer = SentimentIntensityAnalyzer()

    global _sentiment_tokenizer, _sentiment_pipeline
    if model_path is None:
        model_path = _sentiment_model_path
    _sentiment_tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    _sentiment_pipeline = TextClassificationPipeline(
        model=model, tokenizer=_sentiment_tokenizer
    )

@register_metric_udf(col_type=String)
def sentiment_nltk(text: str) -> float:
    if _sentiment_analyzer is None:
        raise ValueError(
            "sentiment metrics must initialize sentiment analyzer before evaluation!"
        )
    sentiment_score = _sentiment_analyzer.polarity_scores(text)
    return sentiment_score["compound"]



@register_metric_udf(col_type=String)
def sentiment(text: str) -> float:
    if _sentiment_pipeline is None or _sentiment_tokenizer is None:
        raise ValueError("sentiment score must initialize the pipeline first")
    result = _sentiment_pipeline(
        text, truncation=True, max_length=_sentiment_tokenizer.model_max_length, top_k = None
    )
    sentiment_score = (
        result[0]["score"] if result[0]["label"] == "POSITIVE" else 1 - result[0]["score"]
    )
    return sentiment_score


init()


