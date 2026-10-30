####################################################################################################
#                                             icons.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-30                                                                              #
#                                                                                                  #
# Purpose: The app icons `mrsuite-gui --install` gives the GUI, made once from logo.png: the logo  #
#          on a transparent square, as PNG (Linux), ICNS (macOS) and ICO (Windows), into           #
#          mrsuite/gui/static. Run with Pillow: python assets/logo/icons.py                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
STATIC = HERE.parents[1] / "mrsuite" / "gui" / "static"
MARGIN = 0.06                   # of the square's side, around the logo's longer side


#**********#
#   main   #
#**********#
def main() -> None:
    logo = Image.open(HERE / "logo.png").convert("RGBA")
    logo = logo.crop(logo.getbbox())                       # its own transparent border off
    side = round(max(logo.size) / (1 - 2 * MARGIN))
    square = Image.new("RGBA", (side, side))
    square.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2), logo)
    square = square.resize((1024, 1024), Image.LANCZOS)

    square.resize((512, 512), Image.LANCZOS).save(STATIC / "icon.png", optimize=True)
    square.save(STATIC / "icon.icns")
    square.save(STATIC / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])


if __name__ == "__main__":
    main()
