"""Where the causal decoders are up to in their feature cache, and who owns that position"""

import importlib
import inspect
import unittest

import torch.nn as nn

from distvae.utils import cache_cursor

# Every module holding an adapter that forwards a cache cursor. Walked rather than listed block
# by block, so an adapter added later is covered without anyone remembering to add it here.
ADAPTER_MODULES = (
    "distvae.modules.adapters.resnet_adapters",
    "distvae.modules.adapters.midblock_adapters",
    "distvae.modules.adapters.downsampling_adapters",
    "distvae.modules.adapters.upsampling_adapters",
    "distvae.modules.adapters.vae.decoder_adapters",
    "distvae.modules.adapters.vae.encoder_adapters",
    "distvae.modules.adapters.layers.conv_adapters",
)


class TestCacheCursor(unittest.TestCase):

    def test_omitting_a_cursor_gets_a_fresh_one_every_time(self):
        first, second = cache_cursor(None), cache_cursor(None)
        self.assertEqual(first, [0])
        self.assertEqual(second, [0])
        # Identity matters because blocks advance the cursor in place while walking the cache.
        # Sharing one list would make the second decode start where the first stopped.
        self.assertIsNot(first, second)

    def test_a_cursor_handed_in_is_the_one_used(self):
        # A caller walking the cache itself passes its own position in, and gets it back to keep
        # advancing rather than a copy that strands its progress here.
        mine = [7]
        self.assertIs(cache_cursor(mine), mine)

    def test_no_adapter_defaults_a_mutable_argument(self):
        # Python binds one default per function at definition, not per call. A list bound there
        # is a single list for the life of the process, and what that gives is not an error but
        # a video conditioned on the tail of the previous decode.
        for name in ADAPTER_MODULES:
            module = importlib.import_module(name)
            for attribute, value in vars(module).items():
                if not (
                    isinstance(value, type)
                    and issubclass(value, nn.Module)
                    and value.__module__ == module.__name__
                ):
                    continue
                forward = value.__dict__.get("forward")
                if forward is None:
                    continue
                for parameter in inspect.signature(forward).parameters.values():
                    with self.subTest(module=name, adapter=attribute, arg=parameter.name):
                        self.assertNotIsInstance(parameter.default, (list, dict, set))


if __name__ == "__main__":
    unittest.main()
