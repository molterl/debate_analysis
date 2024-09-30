## @file key_word_extractor.py
#  @brief This file contains all the functions to do keyword extraction.
#  @details This script includes functions to create a LightPipeline for text annotation and to initialize a Spark NLP pipeline model.
#  It uses various Spark NLP components and integrates with PySpark for processing.
#  @author Lucas Molter (github: molterl)
#  @date 2024-09-28
#  @copyright
#  All rights reserved. This file is created by Lucas Molter and is the intellectual property of Lucas Molter.

import argparse
import logging
import re

import pandas as pd
import sparknlp
from IPython.core.display import HTML, display
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, struct, udf
from pyspark.sql.types import ArrayType, DataType, StringType
from sparknlp.annotator import *
from sparknlp.base import *


## @brief This function creates a LightPipeline for text annotation.
#  @param text The text to be annotated.
#  @param model The Spark NLP pipeline model.
#  @return The annotated text.
def createLightPipelineForText(text, model):
    light_pipeline = LightPipeline(model)
    result = light_pipeline.fullAnnotate(text)[0]

    return result


## @brief This function initializes a Spark NLP pipeline model.
#  @details This function creates a pipeline model with the following stages:
#  - DocumentAssembler
#  - SentenceDetector
#  - Tokenizer
#  - StopWordsCleaner with the pretrained Portuguese stop words and the possibility to add custom stop words
#  - YakeKeywordExtraction
#  @param spark The Spark session.
#  @return The initialized Spark NLP pipeline model.
def createPipelineModel(spark, top_n=20, custom_stop_words=None):
    document = DocumentAssembler().setInputCol("text").setOutputCol("document")

    sentenceDetector = (
        SentenceDetector().setInputCols("document").setOutputCol("sentence")
    )

    token = (
        Tokenizer()
        .setInputCols("sentence")
        .setOutputCol("token")
        .setContextChars(["(", ")", "?", "!", ".", ","])
    )

    stop_words_cleaner = (
        StopWordsCleaner()
        .pretrained("stopwords_pt", "pt")
        .setInputCols(["token"])
        .setOutputCol("clearedToken")
    )

    if custom_stop_words:
        custom_stop_words.extend(stop_words_cleaner.getStopWords())
        custom_stop_words = list(set(custom_stop_words))
        stop_words_cleaner = stop_words_cleaner.setStopWords(custom_stop_words)

    keywords = (
        YakeKeywordExtraction()
        .setInputCols("clearedToken")
        .setOutputCol("keywords")
        .setMinNGrams(1)
        .setMaxNGrams(5)
        .setNKeywords(top_n)
    )

    pipeline = Pipeline(
        stages=[document, sentenceDetector, token, stop_words_cleaner, keywords]
    )

    initial_dataframe = spark.createDataFrame([[""]]).toDF("text")

    model = pipeline.fit(initial_dataframe)

    return model


## @brief This function extracts the keywords from the LightPipeline and return them as a DataFrame.
#  @param light_pipeline The LightPipeline.
#  @return The keywords.
def getNKeywords(light_pipeline):
    keywords = light_pipeline["keywords"]
    df = pd.DataFrame(
        [
            (
                kw.result,
                kw.begin,
                kw.end,
                float(kw.metadata["score"]),
                int(kw.metadata["sentence"]),
            )
            for kw in keywords
        ],
        columns=["keywords", "begin", "end", "score", "sentence"],
    )
    df = df.drop_duplicates(subset=["keywords"])
    df = df.sort_values(by=["sentence", "score"])
    return df


## @brief This function reads a text file.
#  @param path The path to the text file.
def readTextFile(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


## @brief This function reads a custom stop word file.
#  @param file_path The path to the custom stop word file.
#  @return The custom stop words in a list.
def readCustomStopWord(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        custom_stop_words = [line.strip() for line in file if line.strip()]
    return custom_stop_words


## @brief Main function that initializes the Spark sessions and calls the other functions.
#  @param debate_text_path The path to the debate text file.
def main(debate_text_path, custom_stop_words_path):
    spark = sparknlp.start()
    logging.info("Spark NLP version: %s", sparknlp.version())
    text = readTextFile(debate_text_path)
    custom_stop_words = readCustomStopWord(custom_stop_words_path)
    model = createPipelineModel(spark, custom_stop_words=custom_stop_words)
    light_pipeline = createLightPipelineForText(text, model)
    top_n_keywords = getNKeywords(light_pipeline)
    print(top_n_keywords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract key words from the debate in the path."
    )
    parser.add_argument("debate_path", type=str, help="Path to the input text file")
    parser.add_argument("custom_words", type=str, help="Path to the input text file")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main(args.debate_path, args.custom_words)
