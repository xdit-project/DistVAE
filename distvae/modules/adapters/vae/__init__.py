"""Public VAE adapter exports, loaded only when requested."""

from importlib import import_module


_DECODERS = (
    "DecoderAdapter",
    "HunyuanVideo15DecoderAdapter",
    "HunyuanVideoDecoderAdapter",
    "LTX2VideoDecoderAdapter",
    "QwenImageDecoderAdapter",
    "WanDecoderAdapter",
)
_ENCODERS = (
    "EncoderAdapter",
    "HunyuanVideo15EncoderAdapter",
    "HunyuanVideoEncoderAdapter",
    "LTX2VideoEncoderAdapter",
    "QwenImageEncoderAdapter",
    "WanEncoderAdapter",
)
_EXPORTS = {
    **{name: "decoder_adapters" for name in _DECODERS},
    **{name: "encoder_adapters" for name in _ENCODERS},
}

__all__ = [*_DECODERS, *_ENCODERS]


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
