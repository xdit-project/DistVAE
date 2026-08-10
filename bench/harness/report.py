"""Versioned benchmark records, provenance, rendering, and exit policy."""

import importlib.metadata
import json
import subprocess
from pathlib import Path

import torch

SCHEMA_VERSION = 3


def _version(distribution, module=None):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return (
            getattr(module, "__version__", "unknown")
            if module is not None
            else "unknown"
        )


def _distvae_revision():
    try:
        import distvae

        root = Path(distvae.__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance():
    """Return library versions and the DistVAE source revision when available."""
    import diffusers
    import distvae

    return {
        "versions": {
            "torch": torch.__version__,
            "diffusers": _version("diffusers", diffusers),
            "distvae": _version("distvae", distvae),
        },
        "provenance": {"distvae_git_revision": _distvae_revision()},
    }


def make_record(
    family,
    half,
    shape,
    composition,
    measurement=None,
    error=None,
    *,
    dtype,
    world_size,
):
    """Build one self-contained schema-versioned result."""
    dtype_name = str(dtype).removeprefix("torch.")
    record = {
        "schema_version": SCHEMA_VERSION,
        **provenance(),
        "family": family,
        "half": half,
        "shape": shape,
        "composition": composition,
        "runtime": {"dtype": dtype_name, "world_size": int(world_size)},
        "measurement": measurement or {},
    }
    if error is not None:
        record["error"] = error
    return record


def set_agreement_policy(agreement, tiling_enabled):
    """Record whether the raw agreement verdict controls process success."""
    numerical_difference = agreement.get("disagreement_type") == "numerical"
    agreement["enforced"] = not (tiling_enabled and numerical_difference)
    if tiling_enabled and numerical_difference:
        agreement["measured_not_enforced"] = (
            "tiling changes arithmetic; the measured difference remains reported"
        )


def report_status(records):
    """Return failure only after all records are ready to be written."""
    if any(record is None or "error" in record for record in records):
        return 1
    for record in records:
        agreement = record.get("measurement", {}).get("agreement")
        if agreement and agreement.get("enforced", True) and not agreement["ok"]:
            return 1
    return 0


def write_json(path, records):
    """Write one record as an object and a grid as an array."""
    payload = records[0] if len(records) == 1 else records
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def render(record, half):
    """Render the compact human-readable view of one record."""
    measurement = record.get("measurement", {})
    mode = record.get("composition", {}).get("execution")
    if "error" in record:
        print(
            f"{record['composition'].get('name', 'cell')} failed: "
            f"{record['error']['type']}: {record['error']['message']}",
            flush=True,
        )
        return
    if mode == "describe-only":
        print(json.dumps(measurement["description"], sort_keys=True), flush=True)
        return
    if mode == "tile-shape-costs":
        costs = measurement["tile_shape_costs"]
        print(
            json.dumps(
                {
                    "latent_window": costs["latent_window"],
                    "analysis": costs["analysis"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    collectives = measurement["collectives"]
    timing = measurement["timing"]
    print(f"\n--- collectives per {half} call ---", flush=True)
    for name, entry in collectives["by_call"].items():
        maximum = collectives["by_call_max"].get(name, entry["calls"])
        print(
            f"  {name:<24} {entry['calls']:>6} calls  "
            f"{maximum:>6} max  {entry['bytes'] / 1e6:>10.2f} MB",
            flush=True,
        )
    print(
        f"median {timing['median_s'] * 1000:.1f} ms   "
        f"peak {measurement['peak_vram_mb']:.0f} MB",
        flush=True,
    )
    agreement = measurement.get("agreement")
    if agreement is not None:
        verdict = "matches" if agreement["ok"] else "differs from"
        print(f"output {verdict} the unsharded reference: {agreement}", flush=True)
