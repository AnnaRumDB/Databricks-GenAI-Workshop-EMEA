# Databricks notebook source
# MAGIC %pip install mlflow[genai]>=2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ##Adding API to use external models in model serving 

# COMMAND ----------

# MAGIC %md More info here https://docs.databricks.com/en/generative-ai/external-models/index.html

# COMMAND ----------

#!Uncomment if you want to add your own OpenAI API key!

#KEY_NAME = "openai_api_key" #Use the key name you prefer
#SCOPE_NAME = "anyar" #Use the scope name you prefer
#OPENAI_API_KEY = ""

# COMMAND ----------

#!Uncomment if you want to add your own OpenAI API key!

#import time
#from databricks.sdk import WorkspaceClient
#w = WorkspaceClient()
#w.secrets.create_scope("myscope")
#w.secrets.put_secret(scope=SCOPE_NAME, key=KEY_NAME, string_value=OPENAI_API_KEY)
#w.secrets.list_secrets(scope=SCOPE_NAME)

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
client.create_endpoint(
    name="openai-chat-endpoint",
    config={
        "served_entities": [{
            "external_model": {
                "name": "gpt-3.5-turbo",
                "provider": "openai",
                "task": "llm/v1/chat",
                "openai_config": {
                    "openai_api_key": "{{secrets/anyar/openai_api_key}}" #change to your own scope and key names 
                }
            }
        }]
    }
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Using Foundation Models API to query Mixtral 8x7b

# COMMAND ----------

# MAGIC %md More info here https://www.databricks.com/blog/build-genai-apps-faster-new-foundation-model-capabilities

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_query(
# MAGIC     'databricks-mixtral-8x7b-instruct',
# MAGIC     'Describe Databricks SQL in 30 words.'
# MAGIC   ) AS chat

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
inputs = {
    "messages": [
        {
            "role": "user",
            "content": "List 3 reasons why you should train an AI model on domain specific data sets? No explanations required."
        }
    ],
    "max_tokens": 150,
    "temperature": 0
}

response = client.predict(endpoint="databricks-mixtral-8x7b-instruct", inputs=inputs)
print(response["choices"][0]['message']['content'])

# COMMAND ----------

inputs = {
  "prompt": "List 3 reasons why you should train an AI model on domain specific data sets? No explanations required.",
  "max_tokens": 128
}

response = client.predict(endpoint="openai-completions-endpoint", inputs=inputs)
print(response["choices"][0]['text'])
