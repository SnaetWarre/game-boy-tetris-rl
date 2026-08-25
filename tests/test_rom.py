import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from gb_tetris_rl.game.rom import (
    RomValidationError,
    read_cartridge_title,
    validate_pandoras_blocks_rom,
)


def create_test_rom(cartridge_title: bytes = b"DMGTRIS") -> bytes:
    rom_bytes = bytearray(0x8000)
    title_start = 0x134
    rom_bytes[title_start : title_start + len(cartridge_title)] = cartridge_title
    return bytes(rom_bytes)


class CartridgeHeaderTests(unittest.TestCase):
    def test_reads_cartridge_title(self) -> None:
        self.assertEqual(read_cartridge_title(create_test_rom()), "DMGTRIS")

    def test_rejects_file_without_a_complete_header(self) -> None:
        with self.assertRaisesRegex(RomValidationError, "too small"):
            read_cartridge_title(b"short")

    def test_validates_source_matched_pandoras_blocks_rom(self) -> None:
        rom_bytes = create_test_rom()
        expected_digest = sha256(rom_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = Path(temporary_directory) / "PandorasBlocks.gbc"
            rom_path.write_bytes(rom_bytes)

            with patch("gb_tetris_rl.game.rom.PANDORAS_BLOCKS_SHA256", expected_digest):
                validated_rom = validate_pandoras_blocks_rom(rom_path)

        self.assertEqual(validated_rom.path, rom_path.resolve())
        self.assertEqual(validated_rom.cartridge_title, "DMGTRIS")

    def test_rejects_an_unknown_pandoras_blocks_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = Path(temporary_directory) / "PandorasBlocks.gbc"
            rom_path.write_bytes(create_test_rom())

            with self.assertRaisesRegex(RomValidationError, "unsupported Pandora's Blocks"):
                validate_pandoras_blocks_rom(rom_path)

    def test_rejects_a_different_game(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = Path(temporary_directory) / "different-game.gb"
            rom_path.write_bytes(create_test_rom(b"TETRIS"))

            with self.assertRaisesRegex(RomValidationError, "Pandora's Blocks"):
                validate_pandoras_blocks_rom(rom_path)


if __name__ == "__main__":
    unittest.main()
