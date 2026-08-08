from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from gb_tetris_rl.roms import PANDORAS_BLOCKS_SHA256

PANDORAS_BLOCKS_COMMIT = "84b61665339ca25fc1db92107d87df4c702fcb5c"
PANDORAS_BLOCKS_BASE_URL = (
    "https://raw.githubusercontent.com/Villadelfia/dmgtris/"
    f"{PANDORAS_BLOCKS_COMMIT}/bin"
)
PANDORAS_BLOCKS_SYMBOLS_SHA256 = (
    "ce075b93c92f52606e8c81d6add5f665cc9d68e2b736e9c4b328dfaf76b15770"
)


@dataclass(frozen=True)
class HomebrewFiles:
    rom_path: Path
    symbols_path: Path


def download_pandoras_blocks(output_directory: str | Path) -> HomebrewFiles:
    resolved_output_directory = Path(output_directory).expanduser().resolve()
    resolved_output_directory.mkdir(parents=True, exist_ok=True)

    rom_path = resolved_output_directory / "PandorasBlocks.gbc"
    symbols_path = resolved_output_directory / "PandorasBlocks.sym"
    _download_verified_file(
        f"{PANDORAS_BLOCKS_BASE_URL}/PandorasBlocks.gbc",
        rom_path,
        PANDORAS_BLOCKS_SHA256,
    )
    _download_verified_file(
        f"{PANDORAS_BLOCKS_BASE_URL}/PandorasBlocks.sym",
        symbols_path,
        PANDORAS_BLOCKS_SYMBOLS_SHA256,
    )
    return HomebrewFiles(rom_path=rom_path, symbols_path=symbols_path)


def _download_verified_file(source_url: str, destination_path: Path, expected_digest: str) -> None:
    with urlopen(source_url, timeout=30) as download_response:  # noqa: S310
        downloaded_bytes = download_response.read()

    actual_digest = sha256(downloaded_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"checksum mismatch for {destination_path.name}: "
            f"expected {expected_digest}, received {actual_digest}"
        )
    destination_path.write_bytes(downloaded_bytes)
