"""Named MiniMind size presets.

minimind-3 is the upstream 64M tutorial model.
minimind-128 is this fork's English product target (16 x 768 = 122.91M).
"""

PRESETS = {
    "minimind-3": {"hidden_size": 768, "num_hidden_layers": 8},
    "minimind-128": {"hidden_size": 768, "num_hidden_layers": 16},
}


def apply_preset(args):
    """Overwrite hidden_size / num_hidden_layers when --preset is set."""
    name = getattr(args, "preset", None)
    if not name:
        return args
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise SystemExit(f"Unknown --preset {name!r}. Choose one of: {known}")
    spec = PRESETS[name]
    args.hidden_size = spec["hidden_size"]
    args.num_hidden_layers = spec["num_hidden_layers"]
    return args
