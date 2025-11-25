#!/bin/bash
# Fix CodeBuild connection access issue

set -e

PROJECT_NAME="tts-service-build"
REGION="ap-south-1"
CONNECTION_ARN="arn:aws:codeconnections:ap-south-1:144631843160:connection/988c5dd5-4b4d-48d1-bda8-792ea825170b"

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
  echo "   Or project doesn't have a service role configured."
  exit 1
fi

# Extract role name from ARN (format: arn:aws:iam::ACCOUNT:role/ROLE_NAME)
ROLE_NAME=$(echo "$SERVICE_ROLE" | sed 's/.*\///')
echo "✅ Found service role: $ROLE_NAME"
echo "   Full ARN: $SERVICE_ROLE"
echo ""

# Check if policy already exists
echo "🔍 Checking existing policies..."
EXISTING=$(aws iam get-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name CodeStarConnectionAccess \
  --region "$REGION" 2>/dev/null || echo "")

if [ -n "$EXISTING" ]; then
  echo "⚠️  Policy already exists. Updating..."
fi

# Create policy document
cat > /tmp/connection-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "codestar-connections:UseConnection"
      ],
      "Resource": "$CONNECTION_ARN"
    }
  ]
}
EOF

echo "📝 Attaching policy to role..."
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name CodeStarConnectionAccess \
  --policy-document file:///tmp/connection-policy.json

echo ""
echo "✅ Policy attached successfully!"
echo ""
echo "🔄 Next steps:"
echo "1. Re-authorize connection in AWS Console:"
echo "   https://console.aws.amazon.com/codesuite/settings/connections?region=$REGION"
echo "2. Click 'Update pending connection' on connection: flocko-sharm173"
echo "3. Authorize in GitHub"
echo "4. Retry CodeBuild:"
echo "   aws codebuild start-build --project-name $PROJECT_NAME --region $REGION"
echo ""

