from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

GAME_BOY_HEADER_MINIMUM_SIZE = 0x150
CARTRIDGE_TITLE_START = 0x134
CARTRIDGE_TITLE_END = 0x144
PANDORAS_BLOCKS_CARTRIDGE_TITLE = "DMGTRIS"
PANDORAS_BLOCKS_SHA256 = "8507c6bbb140fb28d86bffcb3ef7c85c752abca0d627b55968e7895da43ec57d"


class RomValidationError(ValueError):
    """Raised when a file is not the source-matched Pandora's Blocks ROM."""


@dataclass(frozen=True)
class ValidatedRom:
    path: Path
    cartridge_title: str


def read_cartridge_title(rom_bytes: bytes) -> str:
    if len(rom_bytes) < GAME_BOY_HEADER_MINIMUM_SIZE:
        raise RomValidationError("file is too small to contain a valid Game Boy header")

    title_bytes = rom_bytes[CARTRIDGE_TITLE_START:CARTRIDGE_TITLE_END]
    clean_title_bytes = title_bytes.split(b"\x00", maxsplit=1)[0].rstrip(b" ")
    return clean_title_bytes.decode("ascii", errors="replace")


def validate_pandoras_blocks_rom(rom_path: str | Path) -> ValidatedRom:
    resolved_rom_path = Path(rom_path).expanduser().resolve()
    if not resolved_rom_path.is_file():
        raise RomValidationError(f"ROM file does not exist: {resolved_rom_path}")
    if resolved_rom_path.suffix.lower() not in {".gb", ".gbc"}:
        raise RomValidationError("expected a .gb or .gbc Game Boy ROM file")

    rom_bytes = resolved_rom_path.read_bytes()
    cartridge_title = read_cartridge_title(rom_bytes)
    if cartridge_title != PANDORAS_BLOCKS_CARTRIDGE_TITLE:
        raise RomValidationError(
            f"expected Pandora's Blocks cartridge title "
            f"{PANDORAS_BLOCKS_CARTRIDGE_TITLE!r}, found {cartridge_title!r}"
        )

    rom_digest = sha256(rom_bytes).hexdigest()
    if rom_digest != PANDORAS_BLOCKS_SHA256:
        raise RomValidationError(
            "unsupported Pandora's Blocks build; run 'gb-tetris-rl bootstrap' "
            "to fetch the source-matched build"
        )
    return ValidatedRom(path=resolved_rom_path, cartridge_title=cartridge_title)
