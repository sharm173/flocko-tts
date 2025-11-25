#!/bin/bash
# Optimize CodeBuild cache configuration for faster builds

set -e

PROJECT_NAME="tts-service-build"
REGION="ap-south-1"
ACCOUNT_ID="144631843160"
CACHE_BUCKET="codebuild-cache-${ACCOUNT_ID}-${REGION}"

echo "🔧 Optimizing CodeBuild cache configuration..."
echo ""

# Create S3 bucket for cache if it doesn't exist
echo "📦 Checking S3 cache bucket..."
if ! aws s3 ls "s3://${CACHE_BUCKET}" 2>/dev/null; then
  echo "Creating S3 cache bucket: ${CACHE_BUCKET}"
  aws s3 mb "s3://${CACHE_BUCKET}" --region "${REGION}"
  
  # Add lifecycle policy to clean up old cache
  cat > /tmp/cache-lifecycle.json <<EOF
{
  "Rules": [{
    "ID": "DeleteOldCache",
    "Status": "Enabled",
    "Prefix": "",
    "Expiration": {
      "Days": 30
    }
  }]
}
EOF
  aws s3api put-bucket-lifecycle-configuration \
    --bucket "${CACHE_BUCKET}" \
    --lifecycle-configuration file:///tmp/cache-lifecycle.json
  echo "✅ Cache bucket created with 30-day lifecycle"
else
  echo "✅ Cache bucket already exists"
fi

# Update CodeBuild project to use S3 cache
echo ""
echo "📝 Updating CodeBuild project to use S3 cache..."
CACHE_CONFIG=$(cat <<EOF
{
  "type": "S3",
  "location": "${CACHE_BUCKET}",
  "modes": ["LOCAL_DOCKER_LAYER_CACHE", "LOCAL_SOURCE_CACHE"]
}
EOF
)

aws codebuild update-project \
  --name "$PROJECT_NAME" \
  --region "$REGION" \
  --cache "$CACHE_CONFIG" \
  --output json > /dev/null

echo "✅ CodeBuild cache updated to S3"
echo ""
echo "Verifying cache configuration..."
aws codebuild batch-get-projects \
  --names "$PROJECT_NAME" \
  --region "$REGION" \
  --query 'projects[0].cache' \
  --output json

echo ""
echo "🎉 Cache optimization complete!"
echo ""
echo "Benefits:"
echo "  ✅ S3 cache persists across different CodeBuild instances"
echo "  ✅ Docker layers cached for faster builds"
echo "  ✅ Source code cached for faster git operations"
echo ""
echo "Expected build times:"
echo "  First build: ~25 min (no change)"
echo "  Subsequent builds: ~5-10 min (with cache)"

