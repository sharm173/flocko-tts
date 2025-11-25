# Why Builds Are Still 25 Minutes

## Root Causes

### 1. CodeBuild LOCAL Cache is Ephemeral
- **Problem**: `LOCAL_DOCKER_LAYER_CACHE` only works on the same build instance
- **Reality**: CodeBuild may use different instances each time → cache is lost
- **Solution**: Switch to **S3 cache** (persists across instances)

### 2. Large Base Images Downloaded Every Time
- **Problem**: PyTorch CUDA base images (~4GB) downloaded on every build
- **Solution**: Pre-pull base images in `pre_build` phase (already added)

### 3. Cache-From May Not Work If Layers Changed
- **Problem**: If Dockerfile changed, cache-from won't help
- **Solution**: Ensure Dockerfile layers are stable (already optimized)

## What I've Fixed

### ✅ Updated buildspec.yml
1. Pre-pull base images (`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` and `runtime`)
2. Multiple `--cache-from` sources (previous image + base images)
3. BuildKit progress logging to see cache hits
4. Cache hit reporting

### ✅ Created S3 Cache Script
- `scripts/optimize-codebuild-cache.sh` switches to S3 cache
- S3 cache persists across different CodeBuild instances

## Expected Results After Fix

| Scenario | Before | After S3 Cache |
|----------|--------|---------------|
| First build | ~25 min | ~25 min |
| No changes | ~25 min | **~5-8 min** ✅ |
| Only app.py | ~25 min | **~8-12 min** ✅ |
| Requirements change | ~25 min | **~15-20 min** ✅ |

## Verify Cache is Working

### Check Build Logs
Look for:
- `CACHED [builder 1/8] FROM docker.io/pytorch/pytorch:...`
- `Using cache` messages
- Build summary showing cache hits

### Check Cache Configuration
```bash
aws codebuild batch-get-projects \
  --names tts-service-build \
  --region ap-south-1 \
  --query 'projects[0].cache'
```

Should show:
```json
{
  "type": "S3",
  "location": "codebuild-cache-144631843160-ap-south-1"
}
```

## If Still Slow

### 1. Check if S3 Cache is Actually Being Used
```bash
aws s3 ls s3://codebuild-cache-144631843160-ap-south-1/ --recursive | head -10
```

### 2. Verify Base Images Are Cached
Check build logs for:
- `Pulling pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` (should be fast if cached)
- `CACHED` messages for base image layers

### 3. Consider Using Smaller Base Image
- Current: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` (~4GB)
- Alternative: Build PyTorch from scratch (slower first build, but smaller image)
- **Not recommended** - current approach is better

### 4. Use Larger CodeBuild Instance
- Current: `BUILD_GENERAL1_LARGE` (8 vCPU, 15GB RAM)
- Upgrade to: `BUILD_GENERAL1_2XLARGE` (16 vCPU, 32GB RAM)
- **Cost**: ~2x, **Speed**: ~1.5-2x faster

## Next Steps

1. **Run the optimization script** (if not done):
   ```bash
   ./scripts/optimize-codebuild-cache.sh
   ```

2. **Commit updated buildspec.yml**:
   ```bash
   git add buildspec.yml
   git commit -m "Optimize build caching: pre-pull base images, multiple cache sources"
   git push
   ```

3. **Monitor next build**:
   - Check logs for `CACHED` messages
   - Should see significant time reduction

## Why 25 Minutes is Expected for First Build

Even with perfect caching:
- **Base image pull**: ~2-3 min (4GB)
- **System deps install**: ~1-2 min
- **Python deps install**: ~10-15 min (torch, transformers, etc. are large)
- **chatterbox-tts install**: ~3-5 min
- **Image push**: ~2-3 min
- **Total**: ~20-28 min

**This is normal for first build!** Subsequent builds should be much faster.

