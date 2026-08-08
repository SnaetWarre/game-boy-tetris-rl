from dataclasses import dataclass
from pathlib import Path

GAME_BOY_HEADER_MINIMUM_SIZE = 0x150
CARTRIDGE_TITLE_START = 0x134
CARTRIDGE_TITLE_END = 0x144


class RomValidationError(ValueError):
    """Raised when a supplied file is not the expected Tetris cartridge image."""


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


def validate_tetris_rom(rom_path: str | Path) -> ValidatedRom:
    resolved_rom_path = Path(rom_path).expanduser().resolve()
    if not resolved_rom_path.is_file():
        raise RomValidationError(f"ROM file does not exist: {resolved_rom_path}")
    if resolved_rom_path.suffix.lower() not in {".gb", ".gbc"}:
        raise RomValidationError("expected a .gb or .gbc Game Boy ROM file")

    cartridge_title = read_cartridge_title(resolved_rom_path.read_bytes())
    if cartridge_title != "TETRIS":
        raise RomValidationError(f"expected cartridge title 'TETRIS', found {cartridge_title!r}")
    return ValidatedRom(path=resolved_rom_path, cartridge_title=cartridge_title)
