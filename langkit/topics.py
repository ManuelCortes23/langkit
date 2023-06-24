from whylogs.experimental.core.metrics.udf_metric import register_metric_udf
from whylogs.core.datatypes import String, Optional
from transformers import (
    pipeline,
)
from . import LangKitConfig

lang_config = LangKitConfig()
_topics = lang_config.topics

#Zero-shot classification models
model_path_1 = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
classifier_1 = pipeline("zero-shot-classification", model=model_path_1)


@register_metric_udf(col_type=String)
def closest_topic_zero_shot_model_1(text: str) -> str:
    output = classifier_1(text, _topics, multi_label=False)
    return output["labels"][0]


#Zero-shot classification models
model_path_2 = "facebook/bart-large-mnli"
classifier_2 = pipeline("zero-shot-classification", model=model_path_2)


@register_metric_udf(col_type=String)
def closest_topic_zero_shot_model_2(text: str) -> str:
    output = classifier_2(text, _topics, multi_label=False)
    return output["labels"][0]



def init(topics: Optional[list] = None):
    global _topics
    if topics:
        _topics = topics

init()
