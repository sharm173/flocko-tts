"""TTS microservice using Chatterbox Multilingual TTS."""
# CRITICAL: Set attention implementation BEFORE any transformers imports
# This must happen before transformers models are imported

import os

# SOLUTION 1: Set the correct environment variable (MUST be before transformers import)
# This is the recommended approach - forces all transformers models to use eager attention
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# SOLUTION 2: Monkey-patch the loader as fallback (in case env var doesn't work)
# This intercepts AutoModel.from_pretrained calls and injects attn_implementation="eager"
# We need to patch ALL model classes that transformers might use
try:
    import transformers
    from transformers import (
        AutoModel, 
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
        PreTrainedModel
    )
    # AutoModelForTextToSpeech might not exist in all transformers versions
    try:
        from transformers import AutoModelForTextToSpeech
    except ImportError:
        AutoModelForTextToSpeech = None
    
    # Store original method
    original_from_pretrained = PreTrainedModel.from_pretrained
    
    # Define wrapper that injects attn_implementation="eager"
    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Force eager attention
        kwargs["attn_implementation"] = "eager"
        print(f"🔧 Monkey-patch injecting attn_implementation='eager' for {pretrained_model_name_or_path}", flush=True)
        return original_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
    # Apply the patch to the base class - this should catch all subclasses
    PreTrainedModel.from_pretrained = patched_from_pretrained
    # Also patch specific classes to be safe
    AutoModel.from_pretrained = patched_from_pretrained
    AutoModelForCausalLM.from_pretrained = patched_from_pretrained
    AutoModelForSeq2SeqLM.from_pretrained = patched_from_pretrained
    AutoModelForSpeechSeq2Seq.from_pretrained = patched_from_pretrained
    if hasattr(transformers, 'AutoModelForTextToSpeech'):
        AutoModelForTextToSpeech.from_pretrained = patched_from_pretrained
    
    print("✅ Patched transformers model loaders to force eager attention", flush=True)
except Exception as e:
    # If patching fails, continue anyway - env var should handle it
    print(f"⚠️ Monkey-patch failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    pass

import io
import time
from typing import Optional

try:
    import numpy as np
    import soundfile as sf
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    sf = None

try:
    # Now import chatterbox - the env var and monkey-patch should ensure eager attention
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from transformers import BitsAndBytesConfig
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False
    ChatterboxMultilingualTTS = None
    BitsAndBytesConfig = None

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from structlog import get_logger

# Configure logging
import logging
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

logger = get_logger(__name__)

# Note: SDPA patch is applied earlier (before importing chatterbox) to ensure it takes effect
# before any transformers models are loaded

# Environment configuration
MODEL_ID = os.getenv("MODEL_ID", "chatterbox-multilingual")
LANGUAGE = os.getenv("LANGUAGE", "hi")  # Hindi default
DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"

logger.info("tts_loading_model", model=MODEL_ID, device=DEVICE, use_4bit=USE_4BIT)

# Initialize TTS model
tts_model: Optional[ChatterboxMultilingualTTS] = None
MODEL_LOADING_ENABLED = os.getenv("MODEL_LOADING_ENABLED", "true").lower() == "true"


def load_model() -> None:
    """Load Chatterbox TTS model."""
    global tts_model

    if not CHATTERBOX_AVAILABLE:
        logger.warning("tts_chatterbox_not_available", message="chatterbox-tts not installed - running in stub mode")
        return

    if not MODEL_LOADING_ENABLED:
        logger.warning("tts_model_loading_disabled", message="Model loading disabled for local testing")
        return

    if DEVICE == "cpu":
        logger.warning("tts_no_gpu", message="No GPU detected. Loading on CPU will be slow.")

    try:
        # from_pretrained() only takes device parameter, downloads default model from HuggingFace
        # This will download to HF_HOME if set, otherwise default HuggingFace cache
        logger.info("tts_downloading_model", message="Downloading model from HuggingFace (this may take a few minutes)")
        
        # Convert device string to torch.device if needed
        if isinstance(DEVICE, str):
            device_obj = torch.device(DEVICE)
        else:
            device_obj = DEVICE
        
        # Note: chatterbox-tts from_pretrained() only accepts device parameter
        # It downloads the default model automatically from HuggingFace
        # Full precision loading (quantization not supported by from_pretrained)
        tts_model = ChatterboxMultilingualTTS.from_pretrained(device_obj)
        
        # Configure model to use 'eager' attention instead of 'sdpa' to avoid output_attentions issues
        # This must be done after model loading
        try:
            # Try multiple approaches to set attention implementation to eager
            # Approach 1: Set on main model config
            if hasattr(tts_model, 'model') and hasattr(tts_model.model, 'config'):
                if hasattr(tts_model.model.config, 'attn_implementation'):
                    tts_model.model.config.attn_implementation = 'eager'
                    logger.info("tts_attention_config", location="model.config", message="Set attn_implementation to 'eager'")
            
            # Approach 2: Set on encoder config if it exists
            if hasattr(tts_model, 'model') and hasattr(tts_model.model, 'encoder'):
                if hasattr(tts_model.model.encoder, 'config'):
                    if hasattr(tts_model.model.encoder.config, 'attn_implementation'):
                        tts_model.model.encoder.config.attn_implementation = 'eager'
                        logger.info("tts_attention_config", location="encoder.config", message="Set attn_implementation to 'eager'")
                # Also try setting on encoder layers directly
                if hasattr(tts_model.model.encoder, 'layers'):
                    for i, layer in enumerate(tts_model.model.encoder.layers):
                        if hasattr(layer, 'config') and hasattr(layer.config, 'attn_implementation'):
                            layer.config.attn_implementation = 'eager'
                    logger.info("tts_attention_config", location="encoder.layers", message=f"Set attn_implementation to 'eager' on {len(tts_model.model.encoder.layers)} layers")
            
            # Approach 3: Set on decoder if it exists
            if hasattr(tts_model, 'model') and hasattr(tts_model.model, 'decoder'):
                if hasattr(tts_model.model.decoder, 'config'):
                    if hasattr(tts_model.model.decoder.config, 'attn_implementation'):
                        tts_model.model.decoder.config.attn_implementation = 'eager'
                        logger.info("tts_attention_config", location="decoder.config", message="Set attn_implementation to 'eager'")
            
            # Approach 4: Set via model's _attn_implementation attribute if it exists
            if hasattr(tts_model, 'model') and hasattr(tts_model.model, '_attn_implementation'):
                tts_model.model._attn_implementation = 'eager'
                logger.info("tts_attention_config", location="model._attn_implementation", message="Set _attn_implementation to 'eager'")
            
            logger.info("tts_attention_config_complete", message="Attempted to set attention implementation to 'eager'")
        except Exception as e:
            logger.warning("tts_attention_config_failed", error=str(e), message="Could not set attention implementation, may use default")

        logger.info("tts_model_loaded", model=MODEL_ID, device=DEVICE)
    except Exception as e:
        logger.error("tts_model_load_failed", error=str(e))
        # Don't raise - allow service to start without model for local testing
        if os.getenv("REQUIRE_MODEL", "false").lower() == "true":
            raise


app = FastAPI(title=f"TTS Service ({MODEL_ID})")


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    load_model()


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe."""
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.get("/readyz")
async def readyz() -> dict:
    """Readiness probe."""
    if tts_model is None and MODEL_LOADING_ENABLED:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    return {"status": "ready", "model": MODEL_ID, "model_loaded": tts_model is not None}


class TTSRequest(BaseModel):
    """Request model for TTS synthesis."""

    text: str
    speaker: Optional[str] = "default"
    speed: float = 1.0
    language: Optional[str] = None


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    speaker: Optional[str] = Form("default"),
    speed: Optional[float] = Form(1.0),
):
    """
    Synthesize speech from text.

    Args:
        text: Text to synthesize
        language: Language code (defaults to env LANGUAGE)
        speaker: Speaker identifier
        speed: Speech speed multiplier

    Returns:
        Audio file (WAV format)
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not available")

    start_time = time.time()
    lang = language or LANGUAGE

    logger.info("tts_synthesize_start", text_length=len(text), language=lang)

    try:
        if not NUMPY_AVAILABLE or not sf:
            raise HTTPException(status_code=503, detail="numpy/soundfile not available - install full requirements for synthesis")

        # Generate audio using Chatterbox
        # Note: model.generate() returns audio tensor directly (not tuple)
        audio_tensor = tts_model.generate(
            text=text,
            language_id=lang,  # Use language_id (string) not language
            temperature=0.7,
            exaggeration=0.5,
        )
        
        # Convert torch tensor to numpy if needed
        if hasattr(audio_tensor, 'cpu'):
            audio_full = audio_tensor.cpu().numpy()
            # Remove batch dimension if present
            if len(audio_full.shape) > 1:
                audio_full = audio_full.squeeze()
        else:
            audio_full = np.array(audio_tensor)

        # Convert to int16 PCM
        audio_int16 = (audio_full * 32767).astype(np.int16)

        # Convert to WAV bytes
        audio_buf = io.BytesIO()
        sf.write(audio_buf, audio_int16, samplerate=24000, format="WAV")
        audio_bytes = audio_buf.getvalue()

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            "tts_synthesize_success",
            latency_ms=latency_ms,
            text_length=len(text),
            language=lang,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Latency-MS": str(latency_ms),
            },
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("tts_synthesize_error", error=str(e), latency_ms=latency_ms)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@app.post("/v1/tts/stream")
async def stream_tts(request: TTSRequest) -> StreamingResponse:
    """
    Stream TTS audio (chunked response).

    Args:
        request: TTS request with text, speaker, speed, language

    Returns:
        Streaming audio response
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not available")

    start_time = time.time()
    lang = request.language or LANGUAGE

    logger.info("tts_stream_start", text_length=len(request.text), language=lang)

    try:
        if not NUMPY_AVAILABLE:
            raise HTTPException(status_code=503, detail="numpy not available - install full requirements for synthesis")

        # Generate full audio
        # Note: model.generate() returns audio tensor directly (not tuple)
        audio_tensor = tts_model.generate(
            text=request.text,
            language_id=lang,  # Use language_id (string) not language
            temperature=0.7,
            exaggeration=0.5,
        )
        
        # Convert torch tensor to numpy if needed
        if hasattr(audio_tensor, 'cpu'):
            audio_full = audio_tensor.cpu().numpy()
            # Remove batch dimension if present
            if len(audio_full.shape) > 1:
                audio_full = audio_full.squeeze()
        else:
            audio_full = np.array(audio_tensor)

        # Convert to int16 and chunk
        audio_int16 = (audio_full * 32767).astype(np.int16)
        chunk_size = 960  # ~40ms of 24kHz audio

        def generate_chunks():
            """Generate audio chunks for streaming."""
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i : i + chunk_size]
                yield chunk.tobytes()

        latency_ms = (time.time() - start_time) * 1000
        logger.info("tts_stream_success", latency_ms=latency_ms)

        return StreamingResponse(
            content=generate_chunks(),
            media_type="application/octet-stream",
            headers={
                "X-Latency-MS": str(latency_ms),
            },
        )

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("tts_stream_error", error=str(e), latency_ms=latency_ms)
        raise HTTPException(status_code=500, detail=f"TTS streaming failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
