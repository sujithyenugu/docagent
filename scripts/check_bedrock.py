#!/usr/bin/env python3
"""Check if the Anthropic use case form has been approved.

Usage:
    source .venv/bin/activate
    python scripts/check_bedrock.py
"""
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

region = os.getenv("AWS_REGION", "us-east-1")
model = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")

rt = boto3.client("bedrock-runtime", region_name=region)

try:
    rt.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": "ping"}]}],
        inferenceConfig={"maxTokens": 1},
    )
    print(f"APPROVED — {model} is reachable. Run: python scripts/setup_aws.py")
except Exception as e:
    msg = str(e)
    if "use case" in msg.lower():
        print("PENDING — Anthropic use case form not yet approved.")
        print("Check status at: https://console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog")
    elif "throttl" in msg.lower():
        print("APPROVED (throttled) — model is reachable. Run: python scripts/setup_aws.py")
    else:
        print(f"ERROR: {msg}")
