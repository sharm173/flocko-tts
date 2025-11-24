# Scripts

Helper scripts for building, testing, and running the TTS service.

## Available Scripts

- **`run_local.sh`** - Run service locally without model (stub mode)
- **`run_docker.sh`** - Build and run Docker container
- **`test_synthesis.sh`** - Test TTS synthesis endpoints
- **`build_fast.sh`** - Fast Docker build with optimizations
- **`setup_local.sh`** - Setup local development environment
- **`fix-connection.sh`** - Fix CodeBuild GitHub connection issues

## Usage

```bash
# Run locally (no model)
./scripts/run_local.sh

# Build and run Docker
./scripts/run_docker.sh

# Test synthesis
./scripts/test_synthesis.sh
```

