"""
main.py – Application entry point.

  python main.py analyze path/to/audio.wav
  python main.py batch  path/to/directory/
  python main.py record --duration 60
  python main.py serve
  python main.py devices
"""

from interface.cli import cli

if __name__ == "__main__":
    cli()
