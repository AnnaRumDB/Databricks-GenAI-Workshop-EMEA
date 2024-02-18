# Databricks notebook source
# MAGIC %md
# MAGIC # LLM RAG Evaluation with MLflow Example Notebook
# MAGIC
# MAGIC In this notebook, we will demonstrate how to evaluate various a RAG system with MLflow.

# COMMAND ----------

# DBTITLE 1,Code Dependencies Installation
# MAGIC %pip install mlflow>=2.8.1
# MAGIC %pip install openai
# MAGIC %pip install chromadb==0.4.15
# MAGIC %pip install langchain==0.0.344
# MAGIC %pip install tiktoken
# MAGIC %pip install 'mlflow[genai]'
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install langchain
# MAGIC %pip install langchain --upgrade

# COMMAND ----------

# DBTITLE 1,Restart Python Library
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Python Package Import Statement

import os
import pandas as pd
import mlflow
import chromadb
import openai
import langchain

# COMMAND ----------

# check mlflow version
mlflow.__version__

# COMMAND ----------

# check chroma version
chromadb.__version__

# COMMAND ----------

# check langchain version
langchain.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set-up Databricks Workspace Secret

# COMMAND ----------

# DBTITLE 1,Python Databricks Workspace Initialization
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# DBTITLE 1,API Key and Scope for Azure OpenAI
#KEY_NAME = "azureopenai_key"
#SCOPE_NAME = "anyar"
#OPENAI_API_KEY = ""

# COMMAND ----------

# DBTITLE 1,Workspace Secrets Management
import time
from databricks.sdk import WorkspaceClient
#w = WorkspaceClient()
#w.secrets.create_scope("anyar")
#w.secrets.put_secret(scope=SCOPE_NAME, key=KEY_NAME, string_value=OPENAI_API_KEY)
w.secrets.list_secrets(scope="anyar")

# COMMAND ----------

# cleanup
# w.secrets.delete_secret(scope=SCOPE_NAME, key=KEY_NAME)
# w.secrets.delete_scope(scope=SCOPE_NAME)

# COMMAND ----------

#os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope=SCOPE_NAME, key=KEY_NAME)
#os.environ["OPENAI_API_TYPE"] = "azure"
#os.environ["OPENAI_API_VERSION"] = "2023-05-15"
#os.environ["OPENAI_API_BASE"] = "https://openai-for-abe.openai.azure.com/"
#os.environ["OPENAI_DEPLOYMENT_NAME"] = "gpt-35-turbo"
#os.environ["OPENAI_ENGINE"] = "gpt-35-turbo"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create and Test Endpoint on MLflow for OpenAI

# COMMAND ----------

# DBTITLE 1,Dynamic Endpoint Configuration
import mlflow
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

endpoint_name = f"test-endpoint-anyar-demo"
client.create_endpoint(
    name="test-endpoint-anyar-demo",
    config={
        "served_entities": [{
            "external_model": {
                "name": "gpt-3.5-turbo-instruct",
                "provider": "openai",
                "task": "llm/v1/completions",
                "openai_config": {
                    "openai_api_key": "{{secrets/anyar/openai_api_key}}"
                }
            }
        }]
    }
)

# COMMAND ----------

print(client.predict(
    endpoint=endpoint_name,
    inputs={
        "prompt": "How is Pi calculated? Be very concise.",
        "max_tokens": 100,
    }
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create RAG POC with LangChain and log with MLflow
# MAGIC
# MAGIC Use Langchain and Chroma to create a RAG system that answers questions based on the MLflow documentation.

# COMMAND ----------

# extra_params={"temperature": 0.1,
                # "top_p": 0.1,
                # "max_tokens": 500,
                # } #parameters used in AI Playground

# COMMAND ----------

# DBTITLE 1,Code Name: MLFlow Documentation Retrieval QA
import os
import pandas as pd
import mlflow
import chromadb
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.llms import Databricks
from langchain.embeddings.databricks import DatabricksEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

loader = WebBaseLoader(
    [ 
     "https://mlflow.org/docs/latest/index.html",
     "https://mlflow.org/docs/latest/tracking/autolog.html", 
     "https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html",
     "https://mlflow.org/docs/latest/python_api/mlflow.deployments.html" ])

documents = loader.load()
CHUNK_SIZE = 1000
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

llm = Databricks(
    endpoint_name="test-endpoint-anyar-demo",
    
)


# create the embedding function using Databricks Foundation Model APIs
embedding_function = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
docsearch = Chroma.from_documents(texts, embedding_function)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(fetch_k=3),
    return_source_documents=True,
)


# COMMAND ----------

# check langchain version
langchain.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the Vector Database and Retrieval using `mlflow.evaluate()`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an eval dataset (Golden Dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC We can [leveraging the power of an LLM to generate synthetic data for testing](https://mlflow.org/docs/latest/llms/rag/notebooks/question-generation-retrieval-evaluation.html), offering a creative and efficient alternative. To our readers and customers, we emphasize the importance of crafting a dataset that mirrors the expected inputs and outputs of your RAG application. It's a journey worth taking for the incredible insights you'll gain!
# MAGIC

# COMMAND ----------

import ast

EVALUATION_DATASET_PATH = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/llms/RAG/static_evaluation_dataset.csv"

synthetic_eval_data = pd.read_csv(EVALUATION_DATASET_PATH)

# Load the static evaluation dataset from disk and deserialize the source and retrieved doc ids
synthetic_eval_data["source"] = synthetic_eval_data["source"].apply(ast.literal_eval)
synthetic_eval_data["retrieved_doc_ids"] = synthetic_eval_data["retrieved_doc_ids"].apply(ast.literal_eval)

# COMMAND ----------

display(synthetic_eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the Embedding Model with MLflow
# MAGIC You can explore with the full dataset but let's demo with fewer data points

# COMMAND ----------

eval_data = pd.DataFrame(
    {
        "question": [
            "What is MLflow?",
            "What is Databricks?",
            "How to serve a model on Databricks?",
            "How to enable MLflow Autologging for my workspace by default?",
        ],
        "source": [
            ["https://mlflow.org/docs/latest/index.html"],
            ["https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html"],
            ["https://mlflow.org/docs/latest/python_api/mlflow.deployments.html"],
            ["https://mlflow.org/docs/latest/tracking/autolog.html"],
        ],
    }
)


# COMMAND ----------

# DBTITLE 1,Python Embedding Evaluation
from typing import List
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def evaluate_embedding(embedding_function):
    CHUNK_SIZE = 1000
    list_of_documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    docs = text_splitter.split_documents(list_of_documents)
    retriever = Chroma.from_documents(docs, embedding_function).as_retriever()

    def retrieve_doc_ids(question: str) -> List[str]:
        docs = retriever.get_relevant_documents(question)
        doc_ids = [doc.metadata["source"] for doc in docs]
        return doc_ids

    def retriever_model_function(question_df: pd.DataFrame) -> pd.Series:
        return question_df["question"].apply(retrieve_doc_ids)

    with mlflow.start_run() as run:
        evaluate_results = mlflow.evaluate(
                model=retriever_model_function,
                data=eval_data,
                model_type="retriever",
                targets="source",
                evaluators="default",
            )
    return evaluate_results

result1 = evaluate_embedding(DatabricksEmbeddings(endpoint="databricks-bge-large-en"))	
#result2 = evaluate_embedding(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

eval_results_of_retriever_df_bge = result1.tables["eval_results_table"]
#eval_results_of_retriever_df_MiniLM = result2.tables["eval_results_table"]
display(eval_results_of_retriever_df_bge)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate different Top K strategy with MLflow

# COMMAND ----------

# DBTITLE 1,MLFlow Evaluate and Display
with mlflow.start_run() as run:
        evaluate_results = mlflow.evaluate(
        data=eval_results_of_retriever_df_bge,
        targets="source",
        predictions="outputs",
        evaluators="default",
        extra_metrics=[
            mlflow.metrics.precision_at_k(1),
            mlflow.metrics.precision_at_k(2),
            mlflow.metrics.precision_at_k(3),
            mlflow.metrics.recall_at_k(1),
            mlflow.metrics.recall_at_k(2),
            mlflow.metrics.recall_at_k(3),
            mlflow.metrics.ndcg_at_k(1),
            mlflow.metrics.ndcg_at_k(2),
            mlflow.metrics.ndcg_at_k(3),
        ],
    )

display(evaluate_results.tables["eval_results_table"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the Chunking Strategy with MLflow

# COMMAND ----------

# DBTITLE 1,Engineer's Code Evaluator
from typing import List

def evaluate_chunk_size(chunk_size):
  list_of_documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=0)
  docs = text_splitter.split_documents(list_of_documents)
  embedding_function = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
  retriever = Chroma.from_documents(docs, embedding_function).as_retriever()
  
  def retrieve_doc_ids(question: str) -> List[str]:
    docs = retriever.get_relevant_documents(question)
    doc_ids = [doc.metadata["source"] for doc in docs]
    return doc_ids
   
  def retriever_model_function(question_df: pd.DataFrame) -> pd.Series:
    return question_df["question"].apply(retrieve_doc_ids)

  with mlflow.start_run() as run:
      evaluate_results = mlflow.evaluate(
          model=retriever_model_function,
          data=eval_data,
          model_type="retriever",
          targets="source",
          evaluators="default",
      )
  return evaluate_results

result11 = evaluate_chunk_size(3000)
#result2 = evaluate_chunk_size(1000)


display(result11.tables["eval_results_table"])
#display(result2.tables["eval_results_table"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the RAG system using `mlflow.evaluate()`
# MAGIC Create a simple function that runs each input through the RAG chain

# COMMAND ----------

def model(input_df):
    return input_df["questions"].map(qa).tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an eval dataset (Golden Dataset)

# COMMAND ----------

# DBTITLE 1,Pandas dataframe displaying questions.
eval_df = pd.DataFrame(
    {
        "questions": [
            "What is MLflow?",
            "What is Databricks?",
            "How to serve a model on Databricks?",
            "How to enable MLflow Autologging for my workspace by default?",
        ],
    }
)
display(eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate using LLM as a Judge and Basic Metric
# MAGIC
# MAGIC Use relevance metric to determine the relevance of the answer and context. There are other metrics you can use too.
# MAGIC

# COMMAND ----------

from mlflow.deployments import set_deployments_target
set_deployments_target("databricks")

# COMMAND ----------

# DBTITLE 1,Code Snippet 0: MLFlow Question Answering Evaluation
from  mlflow.metrics.genai.metric_definitions import relevance

relevance_metric = relevance(model="endpoints:/databricks-llama-2-70b-chat") #You can also use any model you have hosted on Databricks, models from the Marketplace or models in the Foundation model API

with mlflow.start_run():
    results =  mlflow.evaluate(
        model,
        eval_df,
        model_type="question-answering",
        evaluators="default",
        predictions="result",
        extra_metrics=[relevance_metric, mlflow.metrics.latency()],
        evaluator_config={
            "col_mapping": {
                "inputs": "questions",
                "context": "source_documents",
            }
        }
    )
    print(results.metrics)

display(results.tables["eval_results_table"])
