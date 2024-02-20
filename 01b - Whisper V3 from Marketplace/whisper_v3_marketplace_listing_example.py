# Databricks notebook source
# MAGIC %md
# MAGIC # Overview of whisper_v3 models in Databricks Marketplace Listing
# MAGIC
# MAGIC The whisper_v3 models offered in Databricks Marketplace are text-to-text generation models released by OpenAI. They are [MLflow](https://mlflow.org/docs/latest/index.html) models that packages
# MAGIC [Hugging Face’s implementation for whisper_v3 models](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013)
# MAGIC using the [transformers](https://mlflow.org/docs/latest/models.html#transformers-transformers-experimental)
# MAGIC flavor in MLflow.
# MAGIC
# MAGIC **Input:** binary audio data bytes (e.g. flac, mp3, wav)
# MAGIC
# MAGIC **Output:** string containing the generated text
# MAGIC
# MAGIC For example notebooks of using the whisper_v3 model in various use cases on Databricks, please refer to [the Databricks ML example repository](https://github.com/databricks/databricks-ml-examples/tree/master/llm-models/transcription/whisper).

# COMMAND ----------

# MAGIC %md
# MAGIC # Listed Marketplace Models
# MAGIC - `whisper_large_v3`:
# MAGIC   - It packages [Hugging Face’s implementation for the whisper_large_v3 model](https://huggingface.co/openai/whisper-large-v3).
# MAGIC   - It has 2 Billion parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies
# MAGIC To create and query the model serving endpoint, Databricks recommends to install the newest Databricks SDK for Python.

# COMMAND ----------

# Upgrade to use the newest Databricks SDK
%pip install --upgrade databricks-sdk
dbutils.library.restartPython()

# COMMAND ----------

# Get catalog name to use for the model from the marketplace
current_user = spark.sql("SELECT current_user() as username").collect()[0].username
catalog_name = f'catalog_whisper_{current_user.split("@")[0].split(".")[0]}'
dbutils.widgets.text("catalog_name",catalog_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC Now go to Marketplace and find whisper_large_v3 model there. Install model to the catalog with *catalog_name*!
# MAGIC

# COMMAND ----------

# Select the model from the dropdown list
model_names = ['whisper_large_v3']
dbutils.widgets.dropdown("model_name", model_names[0], model_names)

# COMMAND ----------

# Default catalog name when installing the model from Databricks Marketplace.
# Replace with the name of the catalog containing this model
# You can also specify a different model version to load for inference
version = "1"
model_name = dbutils.widgets.get("model_name")
model_uc_path = f"{catalog_name}.models.{model_name}"
endpoint_name = f'{model_name}_marketplace_{current_user.split("@")[0].split(".")[0]}'

# COMMAND ----------

# MAGIC %md
# MAGIC # Usage

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks recommends that you primarily work with this model via Model Serving
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying the model to Model Serving
# MAGIC
# MAGIC You can deploy this model directly to a Databricks Model Serving Endpoint
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).
# MAGIC
# MAGIC Note: Model serving is not supported on GCP. On GCP, Databricks recommends running `Batch inference using Spark`, 
# MAGIC as shown below.
# MAGIC
# MAGIC We recommend the below workload types for each model size:
# MAGIC | Model Name      | Suggested workload type (AWS) | Suggested workload type (AZURE) |
# MAGIC | --------------- | ----------------------------- | ------------------------------- |
# MAGIC | `whisper_large_v3` | GPU_MEDIUM (AWS) | GPU_LARGE (AZURE) |
# MAGIC
# MAGIC You can create the endpoint by clicking the “Serve this model” button above in the model UI. And you can also
# MAGIC create the endpoint with Databricks SDK as following:
# MAGIC

# COMMAND ----------

# Choose the right workload types based on the model size
workload_type = "GPU_MEDIUM"

# COMMAND ----------

import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
w = WorkspaceClient()

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": endpoint_name,
            "model_name": model_uc_path,
            "model_version": version,
            "workload_type": workload_type,
            "workload_size": "Small",
            "scale_to_zero_enabled": "False",
        }
    ]
})
model_details = w.serving_endpoints.create(name=endpoint_name, config=config)
model_details.result(timeout=datetime.timedelta(minutes=30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL transcription using ai_query
# MAGIC
# MAGIC To generate the text using the endpoint, use `ai_query`
# MAGIC to query the Model Serving endpoint. The first parameter should be the
# MAGIC name of the endpoint you previously created for Model Serving. The second
# MAGIC parameter should be a `named_struct` with name `prompt` and value is the 
# MAGIC column name that containing the instruction text. Extra parameters can be added
# MAGIC to the named_struct too. For supported parameters, please refer to [MLFlow AI gateway completion routes](https://mlflow.org/docs/latest/gateway/index.html#completions)
# MAGIC The third and fourth parameters set the return type, so that
# MAGIC `ai_query` can properly parse and structure the output text.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC ai_query(
# MAGIC   <endpoint name>,
# MAGIC   audio_input,
# MAGIC   'returnType',
# MAGIC   'STRING'
# MAGIC ) as transcription
# MAGIC FROM <TABLE>
# MAGIC ```
# MAGIC
# MAGIC You can use `ai_query` in this manner to generate text in
# MAGIC SQL queries or notebooks connected to Databricks SQL Pro or Serverless
# MAGIC SQL Endpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the text by querying the serving endpoint
# MAGIC With the Databricks SDK, you can query the serving endpoint as follows:

# COMMAND ----------

from datasets import load_dataset
import pandas as pd
import base64
import json

from databricks.sdk import WorkspaceClient

dataset = load_dataset("Nexdata/accented_english", split="train")
sample_path = dataset[0]["audio"]["path"]

# Change it to your own input file name
with open(sample_path, 'rb') as audio_file:
    audio_bytes = audio_file.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('ascii')

dataframe_records = [audio_b64]

w = WorkspaceClient()
response = w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=dataframe_records,
)
print(response.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference using Spark
# MAGIC
# MAGIC You can also directly load the model as a Spark UDF and run batch
# MAGIC inference on Databricks compute using Spark. We recommend using a
# MAGIC GPU cluster with Databricks Runtime for Machine Learning version 14.1
# MAGIC or greater.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

catalog_name = "databricks_whisper_v3_models_anya"
transcribe = mlflow.pyfunc.spark_udf(spark, f"models:/{model_uc_path}/{version}", "string")

# COMMAND ----------

import pandas as pd
from datasets import load_dataset
import base64
import json

dataset = load_dataset("Nexdata/accented_english", split="train")
sample_path = dataset[0]["audio"]["path"]

with open(sample_path, 'rb') as audio_file:
    audio_bytes = audio_file.read()
    dataset = pd.DataFrame(pd.Series([audio_bytes]))

df = spark.createDataFrame(dataset)

# You can use the UDF directly on a text column
transcribed_df = df.select(transcribe(df["0"]).alias('transcription'))

# COMMAND ----------

display(transcribed_df)
