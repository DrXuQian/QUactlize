"""Host identity admission for the bounded PPU1.5 probe, not product routing."""

ZW890_NAMES = frozenset(("ZW890", "PPU-ZW890", "ZW-M890P"))


def validate_identity(identity):
    """Admit explicit names; CUDA compatibility capability is not PPU generation.

    The native qz15_device probe has already required one visible device, and
    its source is compiled only for __HGGC_ARCH__150. Keep those gates separate.
    """
    name = identity.get("name", "").strip().upper()
    if name not in ZW890_NAMES or identity.get("warp_size") != 32:
        raise ValueError(
            "this bring-up requires one visible ZW890/ZW-M890P with warp size32; "
            "got " + repr(identity)
        )
