#!/bin/bash
# Script to help fix GitHub connection issues

set -e

CONNECTION_ARN="arn:aws:codeconnections:ap-south-1:144631843160:connection/988c5dd5-4b4d-48d1-bda8-792ea825170b"
REGION="ap-south-1"

echo "🔍 Checking connection status..."
echo ""

# Check connection status
aws codestar-connections get-connection \
  --connection-arn "$CONNECTION_ARN" \
  --region "$REGION" \
  --query 'Connection.{Name:ConnectionName,Status:ConnectionStatus,Provider:ProviderType}' \
  --output table

echo ""
echo "📋 Connection Details:"
CONNECTION_STATUS=$(aws codestar-connections get-connection \
  --connection-arn "$CONNECTION_ARN" \
  --region "$REGION" \
  --query 'Connection.ConnectionStatus' \
  --output text)

if [ "$CONNECTION_STATUS" = "PENDING" ]; then
  echo "⚠️  Connection is PENDING - needs authorization"
  echo ""
  echo "To fix:"
  echo "1. Go to AWS Console → Developer Tools → Settings → Connections"
  echo "2. Find connection: 988c5dd5-4b4d-48d1-bda8-792ea825170b"
  echo "3. Click 'Update pending connection'"
  echo "4. Authorize in GitHub"
  echo "5. Grant repository access"
elif [ "$CONNECTION_STATUS" = "AVAILABLE" ]; then
  echo "✅ Connection is AVAILABLE"
  echo ""
  echo "If webhook creation still fails:"
  echo "1. Check GitHub App permissions (Settings → Integrations → Installed GitHub Apps)"
  echo "2. Ensure repository has Read access"
  echo "3. Try creating webhook again"
else
  echo "❌ Connection status: $CONNECTION_STATUS"
  echo "Consider creating a new connection"
fi

echo ""
echo "🔗 Connection ARN: $CONNECTION_ARN"
echo ""
echo "To view in console:"
echo "https://console.aws.amazon.com/codesuite/settings/connections?region=$REGION"

