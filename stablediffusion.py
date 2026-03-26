import boto3
import json
import base64
import os

prompt = """
Cinematic beach during rainy season, blue sky with dramatic clouds,
ultra realistic lighting, 4k photography
"""

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

payload = {
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": prompt
    },
    "imageGenerationConfig": {
        "numberOfImages": 1,
        "height": 1024,
        "width": 1024,
        "cfgScale": 8
    }
}

response = bedrock.invoke_model(
    modelId="amazon.titan-image-generator-v2:0",
    body=json.dumps(payload),
    contentType="application/json",
    accept="application/json"
)

response_body = json.loads(response["body"].read())

image_base64 = response_body["images"][0]
image_bytes = base64.b64decode(image_base64)

os.makedirs("output", exist_ok=True)

file_path = "output/generated-img.png"
with open(file_path, "wb") as f:
    f.write(image_bytes)

print("Image saved to:", file_path)