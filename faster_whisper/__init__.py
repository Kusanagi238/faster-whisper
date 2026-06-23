import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper.audio import decode_audio
    from faster_whisper.transcribe import BatchedInferencePipeline, WhisperModel
    from faster_whisper.utils import available_models, download_model, format_timestamp
    from faster_whisper.version import __version__

_lazy_modules = {
    "decode_audio": ("faster_whisper.audio", "decode_audio"),
    "BatchedInferencePipeline": (
        "faster_whisper.transcribe",
        "BatchedInferencePipeline",
    ),
    "WhisperModel": ("faster_whisper.transcribe", "WhisperModel"),
    "available_models": ("faster_whisper.utils", "available_models"),
    "download_model": ("faster_whisper.utils", "download_model"),
    "format_timestamp": ("faster_whisper.utils", "format_timestamp"),
    "__version__": ("faster_whisper.version", "__version__"),
}


def __getattr__(name):
    if name in _lazy_modules:
        module_name, attr = _lazy_modules[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "available_models",
    "decode_audio",
    "WhisperModel",
    "BatchedInferencePipeline",
    "download_model",
    "format_timestamp",
    "__version__",
]
