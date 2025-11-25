#!/bin/bash
# Quick test script to apply SDPA fix in running pod without rebuilding
# Usage: ./scripts/test_in_pod.sh

set -e

POD_NAME=$(kubectl get pod -n voice-stack -l app=tts-service --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "❌ No running pod found"
    exit 1
fi

echo "📦 Testing SDPA fix in pod: $POD_NAME"
echo ""

# Step 1: Copy patched app.py to pod
echo "1️⃣ Copying patched app.py to pod..."
kubectl cp tts-service/app.py voice-stack/$POD_NAME:/app/app.py
echo "✅ File copied"
echo ""

# Step 2: Restart uvicorn
echo "2️⃣ Restarting uvicorn process..."
kubectl exec -n voice-stack $POD_NAME -- pkill -f uvicorn || true
echo "✅ Process killed (will restart automatically)"
echo ""

# Step 3: Wait a moment and check logs
echo "3️⃣ Waiting for restart..."
sleep 5

echo "4️⃣ Checking logs for model load..."
kubectl logs -n voice-stack $POD_NAME --tail=50 | grep -E "(tts_|SDPA|model.*loaded|ERROR|error)" | tail -20

echo ""
echo "✅ Done! Check if model loaded successfully:"
echo "   kubectl logs -n voice-stack $POD_NAME -f"
echo ""
echo "Or test the endpoint:"
echo "   kubectl exec -n voice-stack $POD_NAME -- curl -s http://localhost:8000/readyz"

