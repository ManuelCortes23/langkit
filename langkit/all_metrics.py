from whylogs.experimental.core.udf_schema import udf_schema
from whylogs.core.schema import DeclarativeSchema
from whylogs.core.datatypes import String, Optional

from langkit import injections
from langkit import topics
from langkit import regexes
from langkit import sentiment
from langkit import textstat
from langkit import themes
from langkit import toxicity
from langkit import input_output
from langkit import emotions
from langkit import multi_topic_classification
from langkit import irony

def init(topic_list : Optional[list] = None, json_path = None) -> DeclarativeSchema:
    print("This is Manuel's dev branch by the way")
    injections.init()
    topics.init(topic_list)
    regexes.init()
    sentiment.init()
    textstat.init()
    themes.init(theme_file_path=json_path)
    toxicity.init()
    input_output.init()

    #My metrics
    emotions.init()
    multi_topic_classification.init()
    irony.init()

    text_schema = udf_schema()
    return text_schema
