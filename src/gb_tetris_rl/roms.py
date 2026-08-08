from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

GAME_BOY_HEADER_MINIMUM_SIZE = 0x150
CARTRIDGE_TITLE_START = 0x134
CARTRIDGE_TITLE_END = 0x144
PANDORAS_BLOCKS_SHA256 = "8507c6bbb140fb28d86bffcb3ef7c85c752abca0d627b55968e7895da43ec57d"


class SupportedGame(StrEnum):
    NINTENDO_TETRIS = "nintendo-tetris"
    PANDORAS_BLOCKS = "pandoras-blocks"


class RomValidationError(ValueError):
    """Raised when a supplied file is not the expected Tetris cartridge image."""


@dataclass(frozen=True)
class ValidatedRom:
    path: Path
    cartridge_title: str
    game: SupportedGame


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
    if cartridge_title == "TETRIS":
        supported_game = SupportedGame.NINTENDO_TETRIS
    elif cartridge_title == "DMGTRIS":
        rom_digest = sha256(resolved_rom_path.read_bytes()).hexdigest()
        if rom_digest != PANDORAS_BLOCKS_SHA256:
            raise RomValidationError(
                "unsupported Pandora's Blocks build; run 'gb-tetris-rl bootstrap' "
                "to fetch the source-matched build"
            )
        supported_game = SupportedGame.PANDORAS_BLOCKS
    else:
        raise RomValidationError(
            "expected cartridge title 'TETRIS' or 'DMGTRIS', "
            f"found {cartridge_title!r}"
        )
    return ValidatedRom(
        path=resolved_rom_path,
        cartridge_title=cartridge_title,
        game=supported_game,
    )
