#!/bin/bash
# Enable Docker layer caching in CodeBuild project

set -e

PROJECT_NAME="tts-service-build"
REGION="ap-south-1"

echo "🔧 Enabling Docker layer cache for CodeBuild project..."
echo ""

# Create cache config JSON
CACHE_CONFIG='{
  "type": "LOCAL",
  "modes": ["LOCAL_DOCKER_LAYER_CACHE"]
}'

echo "📝 Updating CodeBuild project with cache configuration..."
aws codebuild update-project \
  --name "$PROJECT_NAME" \
  --region "$REGION" \
  --cache "$CACHE_CONFIG" \
  --output json > /dev/null

echo "✅ Cache enabled successfully!"
echo ""
echo "Verifying cache configuration..."
aws codebuild batch-get-projects \
  --names "$PROJECT_NAME" \
  --region "$REGION" \
  --query 'projects[0].cache' \
  --output json

echo ""
echo "🎉 Next builds will use Docker layer caching!"
echo "Expected build time reduction: 25min → 5-10min (for code-only changes)"
