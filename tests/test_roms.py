import tempfile
import unittest
from pathlib import Path

from gb_tetris_rl.roms import RomValidationError, read_cartridge_title, validate_tetris_rom


def create_test_rom(cartridge_title: bytes = b"TETRIS") -> bytes:
    rom_bytes = bytearray(0x8000)
    title_start = 0x134
    rom_bytes[title_start : title_start + len(cartridge_title)] = cartridge_title
    return bytes(rom_bytes)


class CartridgeHeaderTests(unittest.TestCase):
    def test_reads_cartridge_title(self) -> None:
        self.assertEqual(read_cartridge_title(create_test_rom()), "TETRIS")

    def test_rejects_file_without_a_complete_header(self) -> None:
        with self.assertRaisesRegex(RomValidationError, "too small"):
            read_cartridge_title(b"short")

    def test_validates_tetris_rom_without_exposing_rom_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = Path(temporary_directory) / "owned-tetris.gb"
            rom_path.write_bytes(create_test_rom())

            validated_rom = validate_tetris_rom(rom_path)

            self.assertEqual(validated_rom.path, rom_path.resolve())
            self.assertEqual(validated_rom.cartridge_title, "TETRIS")

    def test_rejects_a_different_game(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            rom_path = Path(temporary_directory) / "different-game.gb"
            rom_path.write_bytes(create_test_rom(b"KIRBY DREAM"))

            with self.assertRaisesRegex(RomValidationError, "expected cartridge title"):
                validate_tetris_rom(rom_path)


if __name__ == "__main__":
    unittest.main()
