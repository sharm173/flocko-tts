#!/bin/bash
# Fix CodeBuild ECR permissions

set -e

PROJECT_NAME="tts-service-build"
REGION="ap-south-1"
ACCOUNT_ID="144631843160"
ECR_REPO="tts-service"

echo "🔍 Checking CodeBuild project..."
echo ""

# Get CodeBuild service role
SERVICE_ROLE=$(aws codebuild batch-get-projects \
  --names "$PROJECT_NAME" \
  --region "$REGION" \
  --query 'projects[0].serviceRole' \
  --output text 2>/dev/null || echo "")

if [ -z "$SERVICE_ROLE" ] || [ "$SERVICE_ROLE" = "None" ]; then
  echo "❌ Could not find CodeBuild project: $PROJECT_NAME"
  exit 1
fi

ROLE_NAME=$(echo "$SERVICE_ROLE" | sed 's/.*\///')
echo "✅ Found service role: $ROLE_NAME"
echo ""

# Check environment variables
echo "🔍 Checking environment variables..."
ENV_VARS=$(aws codebuild batch-get-projects \
  --names "$PROJECT_NAME" \
  --region "$REGION" \
  --query 'projects[0].environment.environmentVariables' \
  --output json)

HAS_AWS_ACCOUNT_ID=$(echo "$ENV_VARS" | grep -q "AWS_ACCOUNT_ID" && echo "yes" || echo "no")
HAS_AWS_DEFAULT_REGION=$(echo "$ENV_VARS" | grep -q "AWS_DEFAULT_REGION" && echo "yes" || echo "no")
HAS_IMAGE_REPO_NAME=$(echo "$ENV_VARS" | grep -q "IMAGE_REPO_NAME" && echo "yes" || echo "no")

if [ "$HAS_AWS_ACCOUNT_ID" = "no" ] || [ "$HAS_AWS_DEFAULT_REGION" = "no" ] || [ "$HAS_IMAGE_REPO_NAME" = "no" ]; then
  echo "⚠️  Missing environment variables. Adding them..."
  echo ""
  
  # Get existing vars and add missing ones
  EXISTING_VARS=$(echo "$ENV_VARS" | jq -c '.')
  
  # Add missing vars
  if [ "$HAS_AWS_ACCOUNT_ID" = "no" ]; then
    EXISTING_VARS=$(echo "$EXISTING_VARS" | jq ". + [{\"name\": \"AWS_ACCOUNT_ID\", \"value\": \"$ACCOUNT_ID\", \"type\": \"PLAINTEXT\"}]")
  fi
  
  if [ "$HAS_AWS_DEFAULT_REGION" = "no" ]; then
    EXISTING_VARS=$(echo "$EXISTING_VARS" | jq ". + [{\"name\": \"AWS_DEFAULT_REGION\", \"value\": \"$REGION\", \"type\": \"PLAINTEXT\"}]")
  fi
  
  if [ "$HAS_IMAGE_REPO_NAME" = "no" ]; then
    EXISTING_VARS=$(echo "$EXISTING_VARS" | jq ". + [{\"name\": \"IMAGE_REPO_NAME\", \"value\": \"$ECR_REPO\", \"type\": \"PLAINTEXT\"}]")
  fi
  
  echo "$EXISTING_VARS" > /tmp/env-vars.json
  
  # Update project
  aws codebuild update-project \
    --name "$PROJECT_NAME" \
    --region "$REGION" \
    --environment "environmentVariables=$(cat /tmp/env-vars.json)" \
    --query 'project.name' \
    --output text
  
  echo "✅ Environment variables updated"
else
  echo "✅ Environment variables are set"
fi

echo ""

# Create ECR policy
cat > /tmp/ecr-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "arn:aws:ecr:$REGION:$ACCOUNT_ID:repository/$ECR_REPO"
    }
  ]
}
EOF

echo "📝 Attaching ECR policy to role..."
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name CodeBuildECRAccess \
  --policy-document file:///tmp/ecr-policy.json

echo ""
echo "✅ ECR permissions attached!"
echo ""
echo "📋 Summary:"
echo "  - Service role: $ROLE_NAME"
echo "  - ECR repository: $ECR_REPO"
echo "  - Environment variables: ✅"
echo ""
echo "🔄 Retry CodeBuild:"
echo "   aws codebuild start-build --project-name $PROJECT_NAME --region $REGION"
echo ""

