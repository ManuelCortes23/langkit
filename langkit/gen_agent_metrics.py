from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema
from whylogs.core.datatypes import String, Optional

from langkit import injections
from langkit import topics
from langkit import sentiment
from langkit import themes
from langkit import toxicity
from langkit import input_output
from langkit import emotions
from langkit import multi_topic_classification
from langkit import irony
from langkit import coherence

def init(topic_list : Optional[list] = None, json_path = None) -> DeclarativeSchema:
    print("This is Manuel's dev branch by the way")
    
    injections.init()
    topics.init(topic_list)
    sentiment.init()
    themes.init(theme_file_path=json_path)
    toxicity.init()
    input_output.init()
    emotions.init()
    multi_topic_classification.init()
    irony.init()
    coherence.init()

    text_schema = udf_schema()
    return text_schema
