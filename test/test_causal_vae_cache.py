"""Temporal feature-cache behavior through the public DistVAE VAE API."""

import sys

import pytest
import torch
from distributed_harness import init_gloo, run_distributed

from distvae.vae.parallel import parallelize_decoder, parallelize_encoder

diffusers = pytest.importorskip("diffusers")


FAMILIES = {
    "wan": (
        diffusers.AutoencoderKLWan,
        {},
        "WanEncoderAdapter",
        "WanDecoderAdapter",
    ),
    "qwen-image": (
        getattr(diffusers, "AutoencoderKLQwenImage", None),
        {"attn_scales": []},
        "QwenImageEncoderAdapter",
        "QwenImageDecoderAdapter",
    ),
}


def _cache_changed(before, after):
    for old, new in zip(before, after):
        if old is None or new is None:
            if old is not new:
                return True
        elif isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
            if old.shape != new.shape or not torch.equal(old, new):
                return True
        elif old != new:
            return True
    return False


def _record_cache_calls(module, records):
    original = module.forward

    def recording_forward(*args, **kwargs):
        cache = kwargs["feat_cache"]
        cursor = kwargs["feat_idx"]
        before = [
            value.clone() if isinstance(value, torch.Tensor) else value
            for value in cache
        ]
        record = {
            "cache": cache,
            "cursor": cursor,
            "start": cursor[0],
            "nonempty_before": sum(value is not None for value in cache),
            "first_chunk": kwargs.get("first_chunk"),
        }
        result = original(*args, **kwargs)
        record.update(
            end=cursor[0],
            nonempty_after=sum(value is not None for value in cache),
            mutated=_cache_changed(before, cache),
        )
        records.append(record)
        return result

    module.forward = recording_forward


def _assert_public_chunks(records, cache_size):
    assert len(records) == 2
    assert [record["start"] for record in records] == [0, 0]
    ends = [record["end"] for record in records]
    assert ends == [ends[0], ends[0]], (ends, cache_size)
    assert 0 < ends[0] <= cache_size, (ends, cache_size)
    assert records[0]["cache"] is records[1]["cache"]
    assert records[0]["cursor"] is not records[1]["cursor"]
    assert records[0]["nonempty_before"] == 0
    assert records[0]["nonempty_after"] > 0
    assert records[1]["nonempty_before"] > 0
    assert all(record["mutated"] for record in records)


def _assert_omitted_cursor_sessions(adapter, sample, cache_size, **kwargs):
    outputs = []
    for _ in range(2):
        cache = [None] * cache_size
        outputs.append(adapter(sample.clone(), feat_cache=cache, **kwargs))
        assert any(value is not None for value in cache)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def cache_worker(rank, world_size, family, seed, master_port):
    init_gloo(rank, world_size, master_port)
    try:
        cls, extra, encoder_adapter, decoder_adapter = FAMILIES[family]

        torch.manual_seed(seed)
        vae = cls(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=1,
            **extra,
        ).eval()
        vae.clear_cache()
        encoder_cache_size = vae._enc_conv_num
        decoder_cache_size = vae._conv_num

        assert parallelize_encoder(vae, None) == encoder_adapter
        assert parallelize_decoder(vae, None) == decoder_adapter
        assert len(vae._enc_feat_map) == encoder_cache_size
        assert len(vae._feat_map) == decoder_cache_size

        encoder_calls = []
        decoder_calls = []
        _record_cache_calls(vae.encoder.encoder, encoder_calls)
        _record_cache_calls(vae.decoder.decoder, decoder_calls)

        pixels = torch.randn(1, 3, 5, 32, 32)
        latents = torch.randn(1, 4, 2, 4, 4)
        with torch.no_grad():
            encoded = vae.encode(pixels).latent_dist.parameters
            decoded = vae.decode(latents).sample

        assert encoded.shape == (1, 8, 2, 4, 4)
        assert decoded.shape == (1, 3, 5, 32, 32)
        _assert_public_chunks(encoder_calls, encoder_cache_size)
        _assert_public_chunks(decoder_calls, decoder_cache_size)
        if family == "wan":
            assert [record["first_chunk"] for record in decoder_calls] == [True, False]
        else:
            assert [record["first_chunk"] for record in decoder_calls] == [None, None]

        with torch.no_grad():
            _assert_omitted_cursor_sessions(
                vae.encoder,
                pixels[:, :, :1],
                encoder_cache_size,
            )
            decoder_options = {"first_chunk": True} if family == "wan" else {}
            _assert_omitted_cursor_sessions(
                vae.decoder,
                latents[:, :, :1],
                decoder_cache_size,
                **decoder_options,
            )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.gloo
@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_public_causal_vae_paths_thread_two_chunks_and_isolate_sessions(
    family, master_port, seed=42
):
    if FAMILIES[family][0] is None:
        pytest.skip("installed diffusers has no AutoencoderKLQwenImage")
    run_distributed(cache_worker, 1, (family, seed), master_port)


def test_unavailable_family_skips_before_spawning(monkeypatch):
    family = "unavailable"
    monkeypatch.setitem(FAMILIES, family, (None, {}, "Encoder", "Decoder"))
    spawned = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "run_distributed",
        lambda *args: spawned.append(args),
    )

    with pytest.raises(pytest.skip.Exception):
        test_public_causal_vae_paths_thread_two_chunks_and_isolate_sessions(
            family, master_port=1
        )

    assert spawned == []
