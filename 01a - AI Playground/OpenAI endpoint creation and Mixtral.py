# Databricks notebook source
# MAGIC %pip install mlflow[genai]>=2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
client.create_endpoint(
    name="openai-completions-endpoint",
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
