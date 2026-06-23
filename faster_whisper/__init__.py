from faster_whisper.audio import decode_audio
from faster_whisper.version import __version__

# Lazy/optional imports: avoid failing package import if optional deps are missing
try:
    from faster_whisper.transcribe import BatchedInferencePipeline, WhisperModel
    from faster_whisper.utils import available_models, download_model, format_timestamp
except ImportError:
    BatchedInferencePipeline = None
    WhisperModel = None
    available_models = None
    download_model = None
    format_timestamp = None

__all__ = [
    "available_models",
    "decode_audio",
    "WhisperModel",
    "BatchedInferencePipeline",
    "download_model",
    "format_timestamp",
    "__version__",
]
