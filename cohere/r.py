import json
import boto3

br = boto3.client(service_name='bedrock-runtime')

AVAILABLE_MODEL_IDS = [
    'cohere.command-r-v1:0', # R
    'cohere.command-r-plus-v1:0', # R+
]

modelId = AVAILABLE_MODEL_IDS[0] # R

chat_history = [
    {"role": "USER", "message": "あなたは和訳が得意な AI です。与えたテキストを日本語に訳してください。"},
    {"role": "CHATBOT", "message": "わかりました。"}
]

body = {
    "message": 'You make inference requests to Cohere Command R and Cohere Command R+ models with InvokeModel or InvokeModelWithResponseStream (streaming). You need the model ID for the model that you want to use. To get the model ID, see Amazon Bedrock model IDs.',
    "chat_history": chat_history,
}

response = br.invoke_model_with_response_stream(body=json.dumps(body), modelId=modelId)

for event in response.get("body"):
    bytes = json.loads(event["chunk"]["bytes"])
    chat_history = bytes['chat_history']
    for chat in chat_history:
        print(f'{chat["role"]}「{chat["message"]}」')
print(f'finish reason : {bytes["finish_reason"]}')
print(f'input tokens : {bytes["amazon-bedrock-invocationMetrics"]["inputTokenCount"]}')
print(f'output tokens : {bytes["amazon-bedrock-invocationMetrics"]["outputTokenCount"]}')
