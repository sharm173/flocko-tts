# Build Optimization Guide

## Current Optimizations Applied

### 1. BuildKit Enabled
- `DOCKER_BUILDKIT=1` enables faster, parallel builds
- Better layer caching and dependency resolution

### 2. ECR Layer Caching
- `--cache-from $REPOSITORY_URI:latest` pulls previous image layers
- Reuses cached layers from ECR, significantly reducing build time

### 3. Inline Cache Metadata
- `--build-arg BUILDKIT_INLINE_CACHE=1` embeds cache metadata in image
- Allows subsequent builds to use cached layers

### 4. CodeBuild Local Cache
- Configured in `codebuild.tf` with `LOCAL_DOCKER_LAYER_CACHE`
- Caches Docker layers between builds on the same build instance

## Expected Build Times

| Scenario | Time | Why |
|----------|------|-----|
| **First build** | ~25 min | Downloads base images, installs all dependencies |
| **No changes** | ~5-8 min | Uses all cached layers |
| **Only app.py changes** | ~8-12 min | Rebuilds last layer only |
| **Requirements change** | ~15-20 min | Rebuilds Python dependencies layer |
| **Base image change** | ~25 min | Full rebuild required |

## Verifying Cache is Working

### Check CodeBuild Cache Configuration
```bash
aws codebuild batch-get-projects \
  --names tts-service-build \
  --region ap-south-1 \
  --query 'projects[0].cache'
```

Should show:
```json
{
  "type": "LOCAL",
  "modes": ["LOCAL_DOCKER_LAYER_CACHE"]
}
```

### Check Build Logs for Cache Hits
Look for messages like:
- `CACHED [builder 1/8] FROM docker.io/pytorch/pytorch:...`
- `Using cache` in build output

### Monitor Build Times
- First build after changes: ~25 min (expected)
- Subsequent builds: Should be much faster if cache is working

## Troubleshooting

### If builds are still slow:

1. **Check if cache is enabled in CodeBuild:**
   ```bash
   aws codebuild batch-get-projects --names tts-service-build --region ap-south-1 --query 'projects[0].cache'
   ```

2. **Verify ECR image exists:**
   ```bash
   aws ecr describe-images --repository-name tts-service --region ap-south-1
   ```

3. **Check build logs for cache usage:**
   - Look for `CACHED` messages
   - Check if `--cache-from` is pulling the image

4. **Ensure buildspec uses BuildKit:**
   - Verify `DOCKER_BUILDKIT=1` is set
   - Check `--cache-from` flag is present

## Additional Optimizations

### If still slow, consider:

1. **Use larger CodeBuild instance:**
   - `BUILD_GENERAL1_2XLARGE` (16 vCPU, 32GB RAM)
   - Faster for large dependency installations

2. **Pre-warm base image:**
   - Pull base image in a separate step
   - Store in ECR for faster access

3. **Split Dockerfile stages:**
   - Build dependencies in separate image
   - Copy only needed artifacts

4. **Use Docker layer cache mount:**
   - Cache pip downloads between builds
   - Requires BuildKit cache mounts

## Current Configuration

- ✅ BuildKit enabled
- ✅ ECR cache-from configured
- ✅ CodeBuild local cache enabled
- ✅ Dockerfile optimized for layer caching

Builds should now be significantly faster on subsequent runs!

