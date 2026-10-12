#!/usr/bin/env python3
"""
scripts/setup_aws.py — one-shot Phase 3 AWS setup for DocumentAgent.

Performs all 7 steps from the README without touching the AWS console:
  1. S3 bucket (versioning + public-access block)
  2. Secrets Manager secret (Landing AI key)
  3. IAM roles + inline policies (Lambda + Bedrock KB)
  4. Lambda function (package, upload, create/update)
  5. S3 event notification  → Lambda trigger on */input/*
  6. Bedrock model access check
  7. OpenSearch Serverless collection + Bedrock Knowledge Base + data source

At the end, BEDROCK_KB_ID and BEDROCK_DS_ID are written back to .env.

Usage:
    source .venv/bin/activate
    python scripts/setup_aws.py [--region us-east-1] [--bucket my-bucket-name]
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import boto3
from dotenv import load_dotenv, set_key

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
LAMBDA_DIR = ROOT / "lambda"
INFRA_DIR = ROOT / "infra"

load_dotenv(ENV_FILE)

# ── Console helpers ──────────────────────────────────────────────────────────
def _step(n, label):
    print(f"\n{'─'*60}")
    print(f"  Step {n}/7 — {label}")
    print(f"{'─'*60}")

def info(msg):  print(f"    {msg}")
def ok(msg):    print(f"  ✓ {msg}")
def warn(msg):  print(f"  ! {msg}", file=sys.stderr)


# ── Step 0: AWS CLI install ──────────────────────────────────────────────────
def ensure_aws_cli():
    # Add user-local bin to PATH so no-sudo installs are found
    local_bin = str(Path.home() / ".local" / "bin")
    os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")

    if shutil.which("aws"):
        result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
        ok(f"AWS CLI: {result.stdout.strip() or result.stderr.strip()}")
        return

    system = platform.system()
    if system == "Darwin":
        _install_aws_cli_macos()
    elif system == "Linux":
        _install_aws_cli_linux(local_bin)
    else:
        warn(
            f"Automatic AWS CLI install is not supported on {system}. Install it "
            "manually: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        )
        sys.exit(1)
    ok("AWS CLI installed.")


def _install_aws_cli_linux(local_bin):
    print("\n[0/7] Installing AWS CLI v2 (Linux, user install, no sudo)...")
    arch = "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
    local_cli = str(Path.home() / ".local" / "aws-cli")
    cmds = [
        f"curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-{arch}.zip -o /tmp/awscliv2.zip",
        "unzip -qo /tmp/awscliv2.zip -d /tmp/awscliv2",
        f"/tmp/awscliv2/aws/install --install-dir {local_cli} --bin-dir {local_bin}",
        "rm -rf /tmp/awscliv2 /tmp/awscliv2.zip",
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True)


def _install_aws_cli_macos():
    # macOS ships AWS CLI v2 as a .pkg, not the Linux zip installer — Homebrew
    # avoids sudo entirely and is the common path on a Mac dev machine.
    if shutil.which("brew"):
        print("\n[0/7] Installing AWS CLI v2 (macOS, via Homebrew)...")
        subprocess.run(["brew", "install", "awscli"], check=True)
        return

    print("\n[0/7] Installing AWS CLI v2 (macOS, official .pkg)...")
    warn("Homebrew not found — falling back to the official installer, which needs an admin password.")
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path = Path(tmpdir) / "AWSCLIV2.pkg"
        subprocess.run(
            ["curl", "-fsSL", "https://awscli.amazonaws.com/AWSCLIV2.pkg", "-o", str(pkg_path)],
            check=True,
        )
        subprocess.run(["sudo", "installer", "-pkg", str(pkg_path), "-target", "/"], check=True)


# ── Step 1: S3 Bucket ────────────────────────────────────────────────────────
def setup_s3(s3, bucket, region):
    _step(1, f"S3 bucket: {bucket}")
    try:
        s3.head_bucket(Bucket=bucket)
        warn("Bucket already exists — skipping create.")
    except s3.exceptions.ClientError:
        kwargs = {"Bucket": bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
        info(f"Created bucket: {bucket}")

    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    ok(f"s3://{bucket} — public access blocked, versioning enabled.")


# ── Step 2: Secrets Manager ──────────────────────────────────────────────────
def setup_secret(sm, landing_ai_key, project):
    _step(2, "Secrets Manager — Landing AI key")
    secret_name = f"{project}/landing-ai-key"
    secret_value = json.dumps({"VISION_AGENT_API_KEY": landing_ai_key})
    try:
        sm.describe_secret(SecretId=secret_name)
        sm.put_secret_value(SecretId=secret_name, SecretString=secret_value)
        warn("Secret already existed — value updated.")
    except sm.exceptions.ResourceNotFoundException:
        sm.create_secret(
            Name=secret_name,
            Description="Landing AI AIDE API key for DocumentAgent Lambda",
            SecretString=secret_value,
        )
        info(f"Created secret: {secret_name}")
    arn = sm.describe_secret(SecretId=secret_name)["ARN"]
    ok(f"Secret ARN: {arn}")
    return arn


# ── Step 3: IAM roles + policies ─────────────────────────────────────────────
def _create_or_get_role(iam, role_name, trust_service):
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": trust_service},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        iam.get_role(RoleName=role_name)
        warn(f"Role {role_name} already exists — skipping create.")
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
        )
        info(f"Created role: {role_name}")
    return iam.get_role(RoleName=role_name)["Role"]["Arn"]


def setup_iam(iam, bucket, region, account_id, project, secret_arn):
    _step(3, "IAM roles and inline policies")

    # ── Bedrock KB role ──
    kb_role = f"AmazonBedrockExecutionRoleForKnowledgeBase_{project}"
    kb_role_arn = _create_or_get_role(iam, kb_role, "bedrock.amazonaws.com")
    kb_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadOutputDocuments",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{bucket}",
                    f"arn:aws:s3:::{bucket}/*/output/*",
                ],
            },
            {
                "Sid": "OpenSearchServerless",
                "Effect": "Allow",
                "Action": ["aoss:APIAccessAll"],
                "Resource": f"arn:aws:aoss:{region}:{account_id}:collection/*",
            },
        ],
    }
    iam.put_role_policy(
        RoleName=kb_role,
        PolicyName=f"{project}-kb-s3-aoss",
        PolicyDocument=json.dumps(kb_policy),
    )
    ok(f"KB role: {kb_role_arn}")

    # ── Lambda role ──
    lambda_role = f"{project}-lambda-role"
    lambda_role_arn = _create_or_get_role(iam, lambda_role, "lambda.amazonaws.com")
    lambda_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadInput",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": f"arn:aws:s3:::{bucket}/*/input/*",
            },
            {
                "Sid": "WriteOutput",
                "Effect": "Allow",
                "Action": ["s3:PutObject"],
                "Resource": f"arn:aws:s3:::{bucket}/*/output/*",
            },
            {
                "Sid": "BedrockSync",
                "Effect": "Allow",
                "Action": ["bedrock:StartIngestionJob"],
                "Resource": f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/*",
            },
            {
                "Sid": "FetchSecret",
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": secret_arn,
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": "arn:aws:logs:*:*:*",
            },
        ],
    }
    iam.put_role_policy(
        RoleName=lambda_role,
        PolicyName=f"{project}-lambda-policy",
        PolicyDocument=json.dumps(lambda_policy),
    )
    ok(f"Lambda role: {lambda_role_arn}")
    return kb_role_arn, lambda_role_arn, lambda_role


# ── Step 4: Lambda function ──────────────────────────────────────────────────
def setup_lambda(lm, bucket, region, account_id, project,
                 lambda_role_arn, secret_arn):
    _step(4, "Lambda function — package and deploy")
    fn_name = f"{project}-ade-processor"

    # Build deployment zip in a temp directory
    info("Packaging Lambda (pip install landingai-ade + handler.py)…")
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / "pkg"
        pkg_dir.mkdir()
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "landingai-ade>=0.1.0", "--quiet", "-t", str(pkg_dir)],
            check=True,
        )
        shutil.copy(LAMBDA_DIR / "handler.py", pkg_dir / "handler.py")

        zip_path = Path(tmpdir) / "function.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in pkg_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(pkg_dir))
        zip_bytes = zip_path.read_bytes()

    info(f"Package size: {len(zip_bytes)/1024/1024:.1f} MB")

    env_vars = {
        # AWS_REGION is reserved by Lambda — it is set automatically to the deploy region
        "BEDROCK_KB_ID": "PLACEHOLDER",   # updated after KB creation
        "BEDROCK_DS_ID": "PLACEHOLDER",
        "LANDING_AI_SECRET_NAME": f"{project}/landing-ai-key",
        "ADE_MODEL": "dpt-2-latest",
    }

    # IAM roles take ~10s to propagate; wait if this is a fresh role
    info("Waiting 12s for IAM role propagation…")
    time.sleep(12)

    try:
        lm.get_function(FunctionName=fn_name)
        warn("Function exists — updating code and config.")
        lm.update_function_code(FunctionName=fn_name, ZipFile=zip_bytes)
        time.sleep(5)
        lm.update_function_configuration(
            FunctionName=fn_name,
            Environment={"Variables": env_vars},
        )
    except lm.exceptions.ResourceNotFoundException:
        lm.create_function(
            FunctionName=fn_name,
            Runtime="python3.12",
            Role=lambda_role_arn,
            Handler="handler.handler",
            Code={"ZipFile": zip_bytes},
            Timeout=300,
            MemorySize=1024,
            Environment={"Variables": env_vars},
        )

    fn_arn = lm.get_function(FunctionName=fn_name)["Configuration"]["FunctionArn"]
    ok(f"Lambda ARN: {fn_arn}")
    return fn_name, fn_arn


# ── Step 5: S3 event notification ───────────────────────────────────────────
def setup_s3_trigger(s3, lm, bucket, fn_name, fn_arn, account_id, region):
    _step(5, "S3 event notification → Lambda trigger on */input/*")

    # Grant S3 permission to invoke Lambda
    try:
        lm.add_permission(
            FunctionName=fn_name,
            StatementId="s3-trigger-input",
            Action="lambda:InvokeFunction",
            Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{bucket}",
            SourceAccount=account_id,
        )
        info("Lambda permission granted to S3 — waiting 10s for propagation…")
        time.sleep(10)
    except lm.exceptions.ResourceConflictException:
        warn("Lambda permission for S3 already exists.")

    # Note: S3 key filters don't support wildcards — the Lambda handler checks
    # the key format (parts[1] == "input") to ignore non-input objects.
    s3.put_bucket_notification_configuration(
        Bucket=bucket,
        NotificationConfiguration={
            "LambdaFunctionConfigurations": [{
                "LambdaFunctionArn": fn_arn,
                "Events": ["s3:ObjectCreated:Put"],
            }]
        },
    )
    ok(f"S3 → Lambda trigger set on s3://{bucket} (handler filters on /input/ prefix)")


# ── Step 6: Bedrock model access ─────────────────────────────────────────────
def check_bedrock_models(session, region):
    _step(6, "Bedrock model access check")
    # Model access is now automatic on first invocation (no manual activation needed).
    # Test both models with lightweight calls to confirm they're reachable.
    rt = session.client("bedrock-runtime", region_name=region)

    # Default agent model: Nova Lite (AWS-native, no Anthropic payment gate)
    # Switch BEDROCK_MODEL_ID to claude-haiku or claude-sonnet for demo quality.
    agent_model = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
    info(f"Testing agent model: {agent_model}…")
    try:
        rt.converse(
            modelId=agent_model,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 1},
        )
        ok(f"  {agent_model}  [reachable]")
    except Exception as e:
        msg = str(e)
        if "use case" in msg.lower() or "payment" in msg.lower():
            warn(f"  {agent_model} blocked: {msg}")
            warn("  Tip: set BEDROCK_MODEL_ID=amazon.nova-lite-v1:0 to bypass Anthropic payment gate.")
            sys.exit(1)
        raise

    # Test embedding model (always Titan Embed v2)
    titan_id = "amazon.titan-embed-text-v2:0"
    info(f"Testing embedding model: {titan_id}…")
    rt.invoke_model(
        modelId=titan_id,
        body=json.dumps({"inputText": "ping", "dimensions": 256}),
        contentType="application/json",
        accept="application/json",
    )
    ok(f"  {titan_id}  [reachable]")


# ── Step 7: OpenSearch Serverless + Bedrock KB ───────────────────────────────
def _aoss_request(session, region, method, url, body=None):
    """Sign and execute an AOSS REST API request using requests-aws4auth."""
    import requests as req_lib
    from requests_aws4auth import AWS4Auth

    frozen = session.get_credentials().get_frozen_credentials()
    auth = AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        "aoss",
        session_token=frozen.token,
    )

    kwargs = {"auth": auth, "timeout": 30}
    if body is not None:
        kwargs["data"] = json.dumps(body)
        kwargs["headers"] = {"Content-Type": "application/json"}

    resp = req_lib.request(method=method, url=url, **kwargs)
    if not resp.ok:
        raise urllib.error.HTTPError(url, resp.status_code, resp.text, {}, None)
    return resp.json()


def setup_knowledge_base(session, region, account_id, bucket, project, kb_role_arn):
    _step(7, "OpenSearch Serverless + Bedrock Knowledge Base")
    aoss = session.client("opensearchserverless", region_name=region)
    bedrock_agent = session.client("bedrock-agent", region_name=region)

    collection_name = f"{project}-vectors"

    # ── Encryption policy (required before collection creation) ──
    # AOSS encryption policy expects an object ({}), not an array ([{}])
    enc_policy = json.dumps({
        "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{collection_name}"]}],
        "AWSOwnedKey": True,
    })
    try:
        aoss.create_security_policy(name=f"{project}-enc", type="encryption", policy=enc_policy)
        info("Encryption policy created.")
    except aoss.exceptions.ConflictException:
        warn("Encryption policy exists.")

    # ── Network policy ──
    net_policy = json.dumps([{
        "Rules": [
            {"ResourceType": "collection", "Resource": [f"collection/{collection_name}"]},
            {"ResourceType": "dashboard", "Resource": [f"collection/{collection_name}"]},
        ],
        "AllowFromPublic": True,
    }])
    try:
        aoss.create_security_policy(name=f"{project}-net", type="network", policy=net_policy)
        info("Network policy created.")
    except aoss.exceptions.ConflictException:
        warn("Network policy exists.")

    # ── Data access policy (KB role + current caller) ──
    caller_arn = session.client("sts").get_caller_identity()["Arn"]
    access_policy = json.dumps([{
        "Rules": [
            {
                "Resource": [f"collection/{collection_name}"],
                "Permission": [
                    "aoss:CreateCollectionItems",
                    "aoss:DeleteCollectionItems",
                    "aoss:UpdateCollectionItems",
                    "aoss:DescribeCollectionItems",
                ],
                "ResourceType": "collection",
            },
            {
                "Resource": [f"index/{collection_name}/*"],
                "Permission": [
                    "aoss:CreateIndex",
                    "aoss:DeleteIndex",
                    "aoss:UpdateIndex",
                    "aoss:DescribeIndex",
                    "aoss:ReadDocument",
                    "aoss:WriteDocument",
                ],
                "ResourceType": "index",
            },
        ],
        "Principal": [kb_role_arn, caller_arn],
    }])
    try:
        aoss.create_access_policy(name=f"{project}-access", type="data", policy=access_policy)
        info("Data access policy created.")
    except aoss.exceptions.ConflictException:
        # Always attempt update — ensures current caller ARN is included after re-runs
        existing = aoss.get_access_policy(
            name=f"{project}-access", type="data"
        )["accessPolicyDetail"]
        try:
            aoss.update_access_policy(
                name=f"{project}-access",
                type="data",
                policy=access_policy,
                policyVersion=existing["policyVersion"],
            )
            info("Data access policy updated with current caller.")
        except Exception as upd_err:
            if "no changes" in str(upd_err).lower():
                info("Data access policy already up to date.")
            else:
                raise
    info("Waiting 20s for access policy propagation…")
    time.sleep(20)

    # ── Create collection ──
    existing = aoss.list_collections(
        collectionFilters={"name": collection_name}
    )["collectionSummaries"]

    if existing:
        coll = existing[0]
        collection_id = coll["id"]
        warn(f"Collection already exists: {collection_id}")
    else:
        resp = aoss.create_collection(name=collection_name, type="VECTORSEARCH")
        collection_id = resp["createCollectionDetail"]["id"]
        info(f"Collection created: {collection_id} — waiting for ACTIVE state…")
        for _ in range(30):
            status = aoss.batch_get_collection(ids=[collection_id])[
                "collectionDetails"
            ][0]["status"]
            if status == "ACTIVE":
                break
            info(f"  status={status}, waiting 15s…")
            time.sleep(15)
        else:
            warn("Collection did not become ACTIVE in time. Re-run after it's ready.")
            sys.exit(1)

    collection_endpoint = aoss.batch_get_collection(ids=[collection_id])[
        "collectionDetails"
    ][0]["collectionEndpoint"]
    collection_arn = aoss.batch_get_collection(ids=[collection_id])[
        "collectionDetails"
    ][0]["arn"]
    ok(f"Collection ACTIVE: {collection_endpoint}")

    # ── Create vector index ──
    index_name = f"{project}-index"
    index_url = f"{collection_endpoint}/{index_name}"
    try:
        _aoss_request(session, region, "GET", index_url)
        warn(f"Index '{index_name}' already exists.")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            _aoss_request(session, region, "PUT", index_url, {
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 1024,  # titan-embed-text-v2:0 default
                            "method": {"engine": "faiss", "name": "hnsw"},
                        },
                        "text": {"type": "text"},
                        "metadata": {"type": "text"},
                    }
                },
            })
            info(f"Vector index '{index_name}' created (dim=1024, engine=faiss).")
            info("Waiting 30s for index to propagate before creating Knowledge Base…")
            time.sleep(30)
        else:
            raise

    # ── Create Bedrock Knowledge Base ──
    embed_model_arn = (
        f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"
    )
    kb_list = bedrock_agent.list_knowledge_bases()["knowledgeBaseSummaries"]
    existing_kb = next((k for k in kb_list if k["name"] == f"{project}-kb"), None)

    if existing_kb:
        kb_id = existing_kb["knowledgeBaseId"]
        warn(f"Knowledge Base already exists: {kb_id}")
    else:
        kb = bedrock_agent.create_knowledge_base(
            name=f"{project}-kb",
            description="DocumentAgent extracted document knowledge base",
            roleArn=kb_role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": embed_model_arn,
                },
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": collection_arn,
                    "vectorIndexName": index_name,
                    "fieldMapping": {
                        "vectorField": "embedding",
                        "textField": "text",
                        "metadataField": "metadata",
                    },
                },
            },
        )
        kb_id = kb["knowledgeBase"]["knowledgeBaseId"]
        info(f"Knowledge Base created: {kb_id}")

    ok(f"Knowledge Base ID: {kb_id}")

    # ── Create data source (S3 output prefix) ──
    ds_list = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)["dataSourceSummaries"]
    existing_ds = next((d for d in ds_list if d["name"] == f"{project}-s3-output"), None)

    if existing_ds:
        ds_id = existing_ds["dataSourceId"]
        warn(f"Data source already exists: {ds_id}")
    else:
        ds = bedrock_agent.create_data_source(
            knowledgeBaseId=kb_id,
            name=f"{project}-s3-output",
            description="ADE-extracted markdown from S3 */output/* prefix",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{bucket}",
                    "inclusionPrefixes": ["output/"],
                },
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": {
                    "chunkingStrategy": "SEMANTIC",
                    "semanticChunkingConfiguration": {
                        "maxTokens": 300,
                        "bufferSize": 1,
                        "breakpointPercentileThreshold": 95,
                    },
                }
            },
        )
        ds_id = ds["dataSource"]["dataSourceId"]
        info(f"Data source created: {ds_id}")

    ok(f"Data Source ID: {ds_id}")
    return kb_id, ds_id


# ── Update Lambda env vars with KB + DS IDs ──────────────────────────────────
def update_lambda_env(lm, fn_name, kb_id, ds_id):
    cfg = lm.get_function_configuration(FunctionName=fn_name)
    env = cfg.get("Environment", {}).get("Variables", {})
    env["BEDROCK_KB_ID"] = kb_id
    env["BEDROCK_DS_ID"] = ds_id
    lm.update_function_configuration(
        FunctionName=fn_name,
        Environment={"Variables": env},
    )
    ok(f"Lambda env updated with BEDROCK_KB_ID={kb_id}, BEDROCK_DS_ID={ds_id}")


# ── Write back to .env ────────────────────────────────────────────────────────
def update_dotenv(kb_id, ds_id, bucket, region):
    set_key(str(ENV_FILE), "S3_STORAGE_ENABLED", "true")
    set_key(str(ENV_FILE), "S3_BUCKET_NAME", bucket)
    set_key(str(ENV_FILE), "AWS_REGION", region)
    set_key(str(ENV_FILE), "BEDROCK_KB_ID", kb_id)
    set_key(str(ENV_FILE), "BEDROCK_DS_ID", ds_id)
    ok(f".env updated with bucket, region, KB ID, DS ID.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DocumentAgent Phase 3 AWS setup")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME", ""))
    args = parser.parse_args()

    region = args.region
    project = "documentagent"

    landing_ai_key = os.getenv("VISION_AGENT_API_KEY", "")
    if not landing_ai_key:
        print("ERROR: VISION_AGENT_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    # ── AWS session ──
    ensure_aws_cli()
    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    print(f"\nAccount: {account_id}  Region: {region}")

    bucket = args.bucket or f"{project}-{account_id[-6:]}"
    print(f"Bucket:  {bucket}")

    s3      = session.client("s3",              region_name=region)
    sm      = session.client("secretsmanager",  region_name=region)
    iam     = session.client("iam")
    lm      = session.client("lambda",          region_name=region)
    bedrock = session.client("bedrock",         region_name=region)

    setup_s3(s3, bucket, region)
    secret_arn = setup_secret(sm, landing_ai_key, project)
    kb_role_arn, lambda_role_arn, lambda_role = setup_iam(
        iam, bucket, region, account_id, project, secret_arn
    )
    fn_name, fn_arn = setup_lambda(
        lm, bucket, region, account_id, project, lambda_role_arn, secret_arn
    )
    setup_s3_trigger(s3, lm, bucket, fn_name, fn_arn, account_id, region)
    check_bedrock_models(session, region)
    kb_id, ds_id = setup_knowledge_base(
        session, region, account_id, bucket, project, kb_role_arn
    )
    update_lambda_env(lm, fn_name, kb_id, ds_id)
    update_dotenv(kb_id, ds_id, bucket, region)

    print(f"\n{'═'*60}")
    print("  DocumentAgent Phase 3 setup complete.")
    print(f"  Bucket:       s3://{bucket}")
    print(f"  Lambda:       {fn_name}")
    print(f"  KB ID:        {kb_id}")
    print(f"  DS ID:        {ds_id}")
    print()
    print("  Upload a document to test:")
    print(f"    aws s3 cp samples/sample_image.jpg s3://{bucket}/<uuid>/input/sample.jpg")
    print()
    print("  Query with Strands Agent:")
    print("    python agent.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
