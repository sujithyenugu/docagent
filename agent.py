"""Strands Agent — DocumentAgent Phase 3.

Queries the Bedrock Knowledge Base (populated by Lambda after ADE extraction)
and answers questions about uploaded documents.

Usage:
  python agent.py                    # interactive chat
  python agent.py "What is the total amount on the invoice?"

Required env vars:
  AWS_REGION        — e.g. us-east-1
  BEDROCK_KB_ID     — Bedrock Knowledge Base ID
  BEDROCK_MODEL_ID  — (optional) defaults to Claude 3.5 Sonnet
"""

import os
import sys

import boto3
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models import BedrockModel

load_dotenv()

_REGION = os.environ["AWS_REGION"]
_KB_ID = os.environ["BEDROCK_KB_ID"]
_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "amazon.nova-lite-v1:0",  # default: free of Anthropic payment gate, low cost
    # for demo: set BEDROCK_MODEL_ID=anthropic.claude-haiku-4-5-20251001-v1:0
)

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=_REGION)


@tool
def query_documents(question: str) -> str:
    """Search the extracted document knowledge base. Returns the most relevant passages."""
    response = _bedrock_runtime.retrieve(
        knowledgeBaseId=_KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": 5}
        },
    )
    results = response.get("retrievalResults", [])
    if not results:
        return "No relevant content found in the knowledge base."
    passages = []
    for r in results:
        text = r["content"]["text"]
        score = r.get("score", 0.0)
        passages.append(f"[relevance={score:.2f}]\n{text}")
    return "\n\n---\n\n".join(passages)


_agent = Agent(
    model=BedrockModel(model_id=_MODEL_ID),
    tools=[query_documents],
    system_prompt=(
        "You are a document assistant. The user asks questions about documents "
        "that have been uploaded and extracted via Landing AI ADE. "
        "Always call query_documents first to retrieve relevant passages, "
        "then synthesise a clear, concise answer grounded in those passages."
    ),
)


def main():
    # Single question from CLI arg, or interactive loop
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = _agent(question)
        print(result)
        return

    print("DocumentAgent — Strands Chat  (Ctrl+C or type 'quit' to exit)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        result = _agent(question)
        print(f"\nAgent: {result}\n")


if __name__ == "__main__":
    main()
