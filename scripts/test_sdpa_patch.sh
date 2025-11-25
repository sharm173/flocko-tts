#!/bin/bash
# Quick test script to validate SDPA patch in running pod
# Usage: ./scripts/test_sdpa_patch.sh

set -e

POD_NAME=$(kubectl get pod -n voice-stack -l app=tts-service --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "❌ No running pod found"
    exit 1
fi

echo "📦 Testing SDPA patch in pod: $POD_NAME"
echo ""

# Test 1: Verify SDPA patch works
echo "1️⃣ Testing SDPA detection..."
kubectl exec -n voice-stack $POD_NAME -- python -c "
import torch
import torch.nn.functional as F
from transformers.utils import is_torch_sdpa_available

print('PyTorch:', torch.__version__)
print('Has SDPA function:', hasattr(F, 'scaled_dot_product_attention'))
print('Transformers check (before):', is_torch_sdpa_available())

# Apply patch
if hasattr(F, 'scaled_dot_product_attention') and not is_torch_sdpa_available():
    import transformers.utils
    def _patched_sdpa_check():
        return hasattr(F, 'scaled_dot_product_attention')
    transformers.utils.is_torch_sdpa_available = _patched_sdpa_check
    print('✅ Patch applied')
    print('Transformers check (after):', is_torch_sdpa_available())
else:
    print('Patch not needed')
" 2>&1

echo ""
echo "2️⃣ Testing model load with patch..."
kubectl exec -n voice-stack $POD_NAME -- python -c "
import sys
import os
os.environ['MODEL_LOADING_ENABLED'] = 'true'

# Apply SDPA patch
import torch
import torch.nn.functional as F
from transformers.utils import is_torch_sdpa_available

if hasattr(F, 'scaled_dot_product_attention') and not is_torch_sdpa_available():
    import transformers.utils
    def _patched_sdpa_check():
        return hasattr(F, 'scaled_dot_product_attention')
    transformers.utils.is_torch_sdpa_available = _patched_sdpa_check

# Try loading model
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model on {device}...')
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    print('✅ Model loaded successfully!')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
" 2>&1 | head -40

