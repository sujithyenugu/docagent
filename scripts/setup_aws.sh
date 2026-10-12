#!/usr/bin/env bash
# scripts/setup_aws.sh — thin wrapper around setup_aws.py
# Usage: bash scripts/setup_aws.sh [--region us-east-1] [--bucket my-bucket]

set -euo pipefail
cd "$(dirname "$0")/.."

# Include user-local AWS CLI install path (no-sudo install lands here)
export PATH="$HOME/.local/bin:$PATH"

# Verify credentials before doing anything
if ! aws sts get-caller-identity &>/dev/null; then
    echo ""
    echo "  AWS credentials not configured. Run one of:"
    echo ""
    echo "  Option A — IAM access keys:"
    echo "    aws configure"
    echo "    # enter Access Key ID, Secret Access Key, region (us-east-1)"
    echo ""
    echo "  Option B — SSO / Identity Center:"
    echo "    aws configure sso"
    echo ""
    echo "  Option C — env vars (CI / temporary):"
    echo "    export AWS_ACCESS_KEY_ID=..."
    echo "    export AWS_SECRET_ACCESS_KEY=..."
    echo "    export AWS_REGION=us-east-1"
    echo ""
    exit 1
fi

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "  Authenticated as account: $ACCOUNT"

# Use project venv if present, else system Python
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# Ensure setup dependencies are available
pip install --quiet boto3 python-dotenv

exec python scripts/setup_aws.py "$@"
