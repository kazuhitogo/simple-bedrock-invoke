# docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

import boto3
import json

br = boto3.client(service_name='bedrock-runtime')

AVAILABLE_MODEL_IDS = [
    'amazon.titan-text-express-v1',
    'amazon.titan-text-lite-v1',
]

modelId = AVAILABLE_MODEL_IDS[0]

phrase = 'Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon via a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don\'t have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with.'
from_lang = '英語'
to_lang = '日本語'

prompt = f'''以下はユーザーと AI のやりとりです。
User: これから{from_lang}の文章を与えるので、{to_lang}に訳してください。ただし、出力は <output> 翻訳結果 </output> としてください。
Bot: わかりました。
User: {phrase}
Bot: <output>
'''

body = json.dumps({
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 8000,
        "stopSequences": ['User:'],
        "temperature": 0,
        "topP": 1
    }
})

response = br.invoke_model(body=body, modelId=modelId, contentType='application/json')
response_body = json.loads(response.get("body").read())
for result in response_body['results']:
    print(f"Token count: {result['tokenCount']}")
    print(f"Output text: {result['outputText']}")
    print(f"Completion reason: {result['completionReason']}")

# 一部文字化けする streaming
# response = br.invoke_model_with_response_stream(body=body, modelId=modelId, contentType='application/json')
# for event in response.get("body"):
#     chunk = json.loads(event["chunk"]["bytes"])
#     output_text = chunk['outputText']
#     print(output_text,end='')
    
# print(f'\ncompletion_reason: {chunk["completionReason"]}')
