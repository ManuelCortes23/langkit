import json
from logging import getLogger
from typing import Optional

from sentence_transformers import util
from torch import Tensor
from whylogs.core.datatypes import String
from whylogs.experimental.core.metrics.udf_metric import register_metric_udf

from langkit.transformer import load_model

from . import LangKitConfig

diagnostic_logger = getLogger(__name__)

_transformer_model = None
_theme_groups = None

#Sentence similarity model
lang_config = LangKitConfig()


def register_theme_udfs():
    if "jailbreaks" in _theme_groups:
        jailbreak_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups["jailbreaks"]
        ]

        @register_metric_udf(col_type=String)
        def jailbreak_similarity(text: str) -> float:
            if _transformer_model is None:
                raise ValueError("Must initialize a transformer before calling encode!")
            similarities = []
            text_embedding = _transformer_model.encode(text, convert_to_tensor=True)
            for embedding in jailbreak_embeddings:
                similarity = get_embeddings_similarity(text_embedding, embedding)
                similarities.append(similarity)
            return max(similarities)

    if "refusals" in _theme_groups:
        refusal_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups["refusals"]
        ]

        @register_metric_udf(col_type=String)
        def refusal_similarity(text: str) -> float:
            if _transformer_model is None:
                raise ValueError("Must initialize a transformer before calling encode!")
            similarities = []
            text_embedding = _transformer_model.encode(text, convert_to_tensor=True)
            for embedding in refusal_embeddings:
                similarity = get_embeddings_similarity(text_embedding, embedding)
                similarities.append(similarity)
            return max(similarities)

    if "workmem" in _theme_groups:
        workmem_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups["workmem"]
        ]

        @register_metric_udf(col_type=String)
        def workmem_similarity(text: str) -> float:
            if _transformer_model is None:
                raise ValueError("Must initialize a transformer before calling encode!")
            similarities = []
            text_embedding = _transformer_model.encode(text, convert_to_tensor=True)
            for embedding in workmem_embeddings:
                similarity = get_embeddings_similarity(text_embedding, embedding)
                similarities.append(similarity)
            return max(similarities)


    if "recentmem" in _theme_groups:
        recentmem_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups["recentmem"]
        ]

        @register_metric_udf(col_type=String)
        def recentmem_similarity(text: str) -> float:
            if _transformer_model is None:
                raise ValueError("Must initialize a transformer before calling encode!")
            similarities = []
            text_embedding = _transformer_model.encode(text, convert_to_tensor=True)
            for embedding in recentmem_embeddings:
                similarity = get_embeddings_similarity(text_embedding, embedding)
                similarities.append(similarity)
            return max(similarities)


    if "longmem" in _theme_groups:
        longmem_embeddings = [
            _transformer_model.encode(s, convert_to_tensor=True)
            for s in _theme_groups["longmem"]
        ]

        @register_metric_udf(col_type=String)
        def longmem_similarity(text: str) -> float:
            if _transformer_model is None:
                raise ValueError("Must initialize a transformer before calling encode!")
            similarities = []
            text_embedding = _transformer_model.encode(text, convert_to_tensor=True)
            for embedding in longmem_embeddings:
                similarity = get_embeddings_similarity(text_embedding, embedding)
                similarities.append(similarity)
            return max(similarities)


def load_themes(json_path: str):
    try:
        skip = False
        with open(json_path, "r", encoding='utf-8') as myfile:
            theme_groups = json.load(myfile)
    except FileNotFoundError:
        skip = True
        diagnostic_logger.warning(f"Could not find {json_path}")
    except json.decoder.JSONDecodeError as json_error:
        skip = True
        diagnostic_logger.warning(f"Could not parse {json_path}: {json_error}")
    if not skip:
        return theme_groups
    return None


def init(transformer_name: Optional[str] = None, theme_file_path: Optional[str] = None):
    global _transformer_model
    global _theme_groups
    if transformer_name is None:
        transformer_name = lang_config.transformer_name
    if theme_file_path is None:
        _theme_groups = load_themes(lang_config.theme_file_path)
    else:
        _theme_groups = load_themes(theme_file_path)

    _transformer_model = load_model(transformer_name)

    register_theme_udfs()


def get_subject_similarity(text: str, comparison_embedding: Tensor) -> float:
    if _transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")
    embedding = _transformer_model.encode(text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding, comparison_embedding)
    return similarity.item()


def get_embeddings_similarity(
    text_embedding: Tensor, comparison_embedding: Tensor
) -> float:
    if _transformer_model is None:
        raise ValueError("Must initialize a transformer before calling encode!")
    similarity = util.pytorch_cos_sim(text_embedding, comparison_embedding)
    return similarity.item()


init()
