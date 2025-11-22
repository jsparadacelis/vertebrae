#!/bin/bash
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
ROLE_NAME=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/)

echo "# Copy these lines EXACTLY into Colab"
echo "import os"
echo ""

# Get the JSON and parse it properly
CREDS=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME)

echo "os.environ['AWS_ACCESS_KEY_ID'] = '$(echo $CREDS | python3 -c "import sys, json; print(json.load(sys.stdin)['AccessKeyId'])")'"
echo "os.environ['AWS_SECRET_ACCESS_KEY'] = '$(echo $CREDS | python3 -c "import sys, json; print(json.load(sys.stdin)['SecretAccessKey'])")'"
echo "os.environ['AWS_SESSION_TOKEN'] = '''$(echo $CREDS | python3 -c "import sys, json; print(json.load(sys.stdin)['Token'])")'''"
echo "os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'"
echo ""
echo "# Expiration: $(echo $CREDS | python3 -c "import sys, json; print(json.load(sys.stdin)['Expiration'])")"
