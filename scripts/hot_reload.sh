#!/bin/bash
# Hot reload app.py in running pod without restarting the whole pod
# Usage: ./scripts/hot_reload.sh

set -e

POD_NAME=$(kubectl get pod -n voice-stack -l app=tts-service --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "❌ No running pod found"
    exit 1
fi

echo "📦 Hot reloading app.py in pod: $POD_NAME"
echo ""

# Step 1: Copy patched app.py
echo "1️⃣ Copying patched app.py to pod..."
kubectl cp tts-service/app.py voice-stack/$POD_NAME:/app/app.py
echo "✅ File copied"
echo ""

# Step 2: Restart uvicorn process (not the whole pod)
echo "2️⃣ Restarting uvicorn process..."
kubectl exec -n voice-stack $POD_NAME -- pkill -f "uvicorn.*app:app" || true
echo "✅ Process killed (will restart automatically)"
echo ""

# Step 3: Wait and check
echo "3️⃣ Waiting for uvicorn to restart..."
sleep 15

echo "4️⃣ Checking if process restarted..."
if kubectl exec -n voice-stack $POD_NAME -- ps aux | grep -q "uvicorn.*app:app"; then
    echo "✅ Uvicorn is running"
else
    echo "⚠️  Uvicorn not found - may need pod restart"
fi

echo ""
echo "5️⃣ Recent logs:"
kubectl logs -n voice-stack $POD_NAME --tail=20 | grep -E "(tts_|model|SDPA|ERROR|error|startup|loaded|PerthNet)" | tail -10 || echo "No relevant logs yet"

echo ""
echo "✅ Hot reload complete!"
echo ""
echo "Monitor with:"
echo "  kubectl logs -n voice-stack $POD_NAME -f"


