import os

# Delay importing faster_whisper until the functions are actually called
# to avoid import-time failures during test collection if optional
# runtime dependencies are missing.


def available_models(*args, **kwargs):
    from faster_whisper import available_models as _available_models

    return _available_models(*args, **kwargs)


def download_model(*args, **kwargs):
    from faster_whisper import download_model as _download_model

    return _download_model(*args, **kwargs)


def test_available_models():
    models = available_models()
    assert isinstance(models, list)
    assert "tiny" in models


def test_download_model(tmpdir):
    output_dir = str(tmpdir.join("model"))

    model_dir = download_model("tiny", output_dir=output_dir)

    assert model_dir == output_dir
    assert os.path.isdir(model_dir)
    assert not os.path.islink(model_dir)

    for filename in os.listdir(model_dir):
        path = os.path.join(model_dir, filename)
        assert not os.path.islink(path)


def test_download_model_in_cache(tmpdir):
    cache_dir = str(tmpdir.join("model"))
    download_model("tiny", cache_dir=cache_dir)
    assert os.path.isdir(cache_dir)
