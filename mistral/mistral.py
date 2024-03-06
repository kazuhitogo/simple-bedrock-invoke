# docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
import json
import boto3

br = boto3.client(service_name='bedrock-runtime',region_name='us-west-2')

AVAILABLE_MODEL_IDS = [
    'mistral.mistral-7b-instruct-v0:2', # 7b
    'mistral.mixtral-8x7b-instruct-v0:1', # 8x7b
]

modelId = AVAILABLE_MODEL_IDS[0] # v2:1

# prompt
phrase = 'Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon via a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don\'t have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with.'
from_lang = '英語'
to_lang = '日本語'
prompt = f'''<s> 以下はユーザーと AI のやりとりです。
[INST] これから{from_lang}の文章を与えるので、{to_lang}に訳してください。ただし、出力は <output> 翻訳結果 </output> としてください。 [/INST]
わかりました。</s> 
[INST] {phrase} [/INST]
<output>'''

body = json.dumps({
    'prompt': prompt,
    'max_tokens': 4096,
    'temperature': 0.7,
    'top_p': 0.7,
    'top_k': 50,
    'stop' : ['</output>']
})

response = br.invoke_model_with_response_stream(body=body, modelId=modelId)
for event in response.get("body"):
    chunk = json.loads(event["chunk"]["bytes"])['outputs'][0]
    print(chunk['text'],end='')
print(f'\nstop_reason: {chunk["stop_reason"]}')
