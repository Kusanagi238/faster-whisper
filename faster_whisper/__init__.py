from faster_whisper.audio import decode_audio
from faster_whisper.version import __version__

# Defer importing submodules that require optional runtime dependencies until they
# are actually used. This avoids importing things like `requests` at package
# import time.


def _get_transcribe():
    from faster_whisper import transcribe as _mod

    return _mod


def _get_utils():
    from faster_whisper import utils as _mod

    return _mod


def AsyncBatchedInferencePipeline(*args, **kwargs):
    return _get_transcribe().AsyncBatchedInferencePipeline(*args, **kwargs)


def BatchedInferencePipeline(*args, **kwargs):
    return _get_transcribe().BatchedInferencePipeline(*args, **kwargs)


def WhisperModel(*args, **kwargs):
    return _get_transcribe().WhisperModel(*args, **kwargs)


def available_models(*args, **kwargs):
    return _get_utils().available_models(*args, **kwargs)


def download_model(*args, **kwargs):
    return _get_utils().download_model(*args, **kwargs)


def format_timestamp(*args, **kwargs):
    return _get_utils().format_timestamp(*args, **kwargs)


__all__ = [
    "available_models",
    "decode_audio",
    "WhisperModel",
    "BatchedInferencePipeline",
    "AsyncBatchedInferencePipeline",
    "download_model",
    "format_timestamp",
    "__version__",
]
