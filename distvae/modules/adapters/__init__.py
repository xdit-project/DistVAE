"""Public adapter exports, loaded only when requested."""

from importlib import import_module


_DOWNSAMPLING = (
    "Downsample2DAdapter",
    "HunyuanVideo15DownBlockAdapter",
    "HunyuanVideo15DownsampleAdapter",
    "HunyuanVideoDownBlockAdapter",
    "HunyuanVideoDownsampleAdapter",
    "LTX2VideoDownBlockAdapter",
    "LTX2VideoDownsamplerAdapter",
    "QwenImageResampleDownAdapter",
    "WanResampleDownAdapter",
    "WanResidualDownBlockAdapter",
)
_UPSAMPLING = (
    "HunyuanVideo15UpBlockAdapter",
    "HunyuanVideo15UpsampleAdapter",
    "HunyuanVideoUpBlockAdapter",
    "HunyuanVideoUpsampleAdapter",
    "LTX2VideoUpBlockAdapter",
    "LTX2VideoUpsamplerAdapter",
    "QwenImageResampleAdapter",
    "QwenImageUpBlockAdapter",
    "Upsample2DAdapter",
    "WanResampleAdapter",
    "WanResidualUpBlockAdapter",
    "WanUpBlockAdapter",
)
_MIDBLOCK = (
    "HunyuanVideo15MidBlockAdapter",
    "HunyuanVideoMidBlockAdapter",
    "LTX2VideoMidBlockAdapter",
    "QwenImageMidBlockAdapter",
    "WanMidBlockAdapter",
)
_RESNET = (
    "HunyuanVideo15ResnetBlockAdapter",
    "HunyuanVideoResnetBlockAdapter",
    "LTX2VideoResnetBlockAdapter",
    "QwenImageResidualBlockAdapter",
    "WanResidualBlockAdapter",
)
_EXPORTS = {
    **{name: "downsampling_adapters" for name in _DOWNSAMPLING},
    **{name: "upsampling_adapters" for name in _UPSAMPLING},
    **{name: "midblock_adapters" for name in _MIDBLOCK},
    **{name: "resnet_adapters" for name in _RESNET},
}

__all__ = [
    *_DOWNSAMPLING,
    *_UPSAMPLING,
    *_MIDBLOCK,
    *_RESNET,
]


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
