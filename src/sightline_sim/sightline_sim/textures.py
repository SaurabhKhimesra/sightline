"""Procedural textures in numpy: floor, tape, labels, signs and screens.

Every function returns an RGB uint8 array (height, width, 3). Nothing here is
measured; colours and patterns are scene choices made to look like a real
assembly hall.
"""

from __future__ import annotations

import numpy as np

# 5 x 7 bitmap font, one string per row.
_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    " ": ["00000"] * 7,
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "|": ["00100"] * 7,
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    "x": ["00000", "00000", "10001", "01010", "00100", "01010", "10001"],
}


def _rgb(c) -> np.ndarray:
    return np.asarray(c, float)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def value_noise(h: int, w: int, cell: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth tileable noise in [0, 1]. h and w should be multiples of cell."""
    gh, gw = max(1, h // cell), max(1, w // cell)
    g = rng.random((gh, gw))
    y = np.arange(h) / cell
    x = np.arange(w) / cell
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    fy = (y - y0)[:, None]
    fx = (x - x0)[None, :]
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    y0m, y1m = y0 % gh, (y0 + 1) % gh
    x0m, x1m = x0 % gw, (x0 + 1) % gw
    a = g[np.ix_(y0m, x0m)]
    b = g[np.ix_(y0m, x1m)]
    c = g[np.ix_(y1m, x0m)]
    d = g[np.ix_(y1m, x1m)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(h: int, w: int, cells=(256, 64, 16, 4), rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for c in cells:
        out += amp * value_noise(h, w, c, rng)
        total += amp
        amp *= 0.55
    return out / total


def box_blur(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return img
    pad = k // 2
    x = np.pad(img, ((pad, pad), (pad, pad)) + ((0, 0),) * (img.ndim - 2), mode="edge")
    c = np.cumsum(np.cumsum(x, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)) + ((0, 0),) * (img.ndim - 2))
    h, w = img.shape[:2]
    return (c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]) / (k * k)


# ---------------------------------------------------------------- surfaces

def epoxy_floor(size: int = 1024, seed: int = 1) -> np.ndarray:
    """Grey-green epoxy coating with soft cloudy variation.

    No fine speckle: at grazing angles it aliased into a dotted pattern.
    """
    rng = np.random.default_rng(seed)
    base = _rgb((0.50, 0.53, 0.52))
    cloud = fbm(size, size, (512, 256, 128), rng) - 0.5
    img = np.ones((size, size, 3)) * base * (1.0 + 0.10 * cloud)[..., None]
    return to_uint8(img)


def hazard_stripes(w: int = 512, h: int = 64, period: int = 64, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    band = ((x + y) // (period // 2)) % 2
    img = np.where(band[..., None] == 0, _rgb((0.96, 0.76, 0.06)), _rgb((0.07, 0.07, 0.07)))
    wear = value_noise(h, w, 16, rng)
    img = img * (0.88 + 0.12 * wear)[..., None]
    return to_uint8(img)


def plain_noise(color, size: int = 256, amount: float = 0.06, cells=(64, 16, 4), seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = fbm(size, size, cells, rng) - 0.5
    img = np.ones((size, size, 3)) * _rgb(color) * (1.0 + 2 * amount * n)[..., None]
    return to_uint8(img)


def brushed(color, size: int = 256, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    streak = box_blur(rng.random((size, size)), 1)
    streak = np.repeat(streak.mean(axis=1, keepdims=True), size, axis=1) * 0.5 + 0.5 * streak
    img = np.ones((size, size, 3)) * _rgb(color) * (0.94 + 0.08 * streak)[..., None]
    return to_uint8(img)


def wall_panels(size: int = 512, seed: int = 5) -> np.ndarray:
    """Insulated sandwich panels: light grey with vertical ribs and dirt near the floor."""
    rng = np.random.default_rng(seed)
    img = np.ones((size, size, 3)) * _rgb((0.80, 0.82, 0.83))
    x = np.arange(size)
    rib = (np.abs(((x % 64) - 32)) < 2).astype(float)
    img *= (1.0 - 0.10 * rib)[None, :, None]
    joint = (x % 256 < 3).astype(float)
    img *= (1.0 - 0.25 * joint)[None, :, None]
    img *= (0.96 + 0.06 * value_noise(size, size, 64, rng))[..., None]
    return to_uint8(img)


def window(size: int = 256) -> np.ndarray:
    """Bright translucent glazing with mullions, used on emissive geoms."""
    img = np.ones((size, size, 3)) * _rgb((0.86, 0.92, 0.98))
    y, x = np.mgrid[0:size, 0:size]
    img[(x % 128) < 6] = _rgb((0.35, 0.38, 0.42))
    img[(y % 128) < 6] = _rgb((0.35, 0.38, 0.42))
    img *= (0.92 + 0.08 * (y / size))[..., None]
    return to_uint8(img)


def kraft(size: int = 256, seed: int = 6) -> np.ndarray:
    """Cardboard carton side: kraft brown, packing tape and a shipping label."""
    rng = np.random.default_rng(seed)
    img = np.ones((size, size, 3)) * _rgb((0.70, 0.54, 0.36))
    img *= (0.93 + 0.10 * fbm(size, size, (64, 16, 4), rng))[..., None]
    img[:, size // 2 - 18:size // 2 + 18] = img[:, size // 2 - 18:size // 2 + 18] * 1.12
    label = np.ones((60, 90, 3)) * 0.95
    for i in range(8):  # barcode bars
        label[36:54, 8 + i * 9:8 + i * 9 + (2 + (i * 7) % 4)] = 0.1
    label[8:14, 8:70] = 0.25
    label[18:24, 8:50] = 0.35
    img[150:210, 140:230] = label
    return to_uint8(np.clip(img, 0, 1))


def esd_mat(size: int = 512, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.ones((size, size, 3)) * _rgb((0.28, 0.38, 0.46))
    img *= (0.95 + 0.08 * fbm(size, size, (128, 32, 8, 2), rng))[..., None]
    return to_uint8(img)


def rack_upright(w: int = 64, h: int = 512) -> np.ndarray:
    """Pallet rack upright face: blue with teardrop holes every 50 mm."""
    img = np.ones((h, w, 3)) * _rgb((0.10, 0.28, 0.58))
    for y in range(8, h, 32):
        img[y:y + 10, w // 2 - 4:w // 2 + 4] = _rgb((0.03, 0.05, 0.08))
    return to_uint8(img)


# ---------------------------------------------------------------- text and screens

def text_mask(text: str, px: int = 4) -> np.ndarray:
    """Float mask (h, w) of text in the 5 x 7 font, px pixels per font pixel, anti-aliased."""
    rows = []
    for line in text.upper().split("\n"):
        glyphs = [_FONT.get(ch, _FONT[" "]) for ch in line]
        cols = []
        for g in glyphs:
            cols.append(np.array([[c == "1" for c in r] for r in g], float))
            cols.append(np.zeros((7, 1)))
        rows.append(np.hstack(cols) if cols else np.zeros((7, 1)))
    width = max(r.shape[1] for r in rows)
    lines = []
    for r in rows:
        lines.append(np.pad(r, ((0, 3), (0, width - r.shape[1]))))
    m = np.vstack(lines)
    big = np.kron(m, np.ones((px * 2, px * 2)))
    big = box_blur(big, 3)
    return big[::2, ::2]


def draw_text(img: np.ndarray, text: str, x: int, y: int, color, px: int = 4) -> None:
    """Draw text into a float RGB image in place, top-left at (x, y)."""
    m = text_mask(text, px)
    h, w = m.shape
    H, W = img.shape[:2]
    h, w = min(h, H - y), min(w, W - x)
    if h <= 0 or w <= 0:
        return
    a = m[:h, :w, None]
    img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + _rgb(color) * a


def label(text: str, w: int = 256, h: int = 96, bg=(0.95, 0.95, 0.93), fg=(0.08, 0.08, 0.08), px: int = 5) -> np.ndarray:
    img = np.ones((h, w, 3)) * _rgb(bg)
    m = text_mask(text, px)
    x = max(0, (w - m.shape[1]) // 2)
    y = max(0, (h - m.shape[0]) // 2)
    draw_text(img, text, x, y, fg, px)
    return to_uint8(img)


def sign(text: str, w: int = 512, h: int = 256, bg=(0.05, 0.30, 0.62), fg=(1.0, 1.0, 1.0), px: int = 6) -> np.ndarray:
    img = np.ones((h, w, 3)) * _rgb(bg)
    img[:10], img[-10:], img[:, :10], img[:, -10:] = 1.0, 1.0, 1.0, 1.0
    m = text_mask(text, px)
    draw_text(img, text, max(0, (w - m.shape[1]) // 2), max(0, (h - m.shape[0]) // 2), fg, px)
    return to_uint8(img)


def hmi_screen(w: int = 1024, h: int = 576) -> np.ndarray:
    """Station display: cell name, the six screw positions and the robot's two camera tiles.

    It shows no counts or times, so a still never displays numbers that were
    not measured.
    """
    img = np.ones((h, w, 3)) * _rgb((0.06, 0.08, 0.11))
    img[:64] = _rgb((0.09, 0.27, 0.50))
    draw_text(img, "SIGHTLINE  CELL 01", 24, 18, (1, 1, 1), 4)
    draw_text(img, "CONTROL BOX", w - 320, 18, (0.8, 0.9, 1.0), 4)
    # left panel: box schematic with the six screw positions
    img[96:520, 24:500] = _rgb((0.11, 0.14, 0.18))
    draw_text(img, "SCREWS", 44, 110, (0.7, 0.8, 0.9), 3)
    img[170:470, 80:440] = _rgb((0.55, 0.58, 0.60))
    img[180:460, 90:430] = _rgb((0.35, 0.38, 0.41))
    yy, xx = np.mgrid[0:h, 0:w]
    for i, (cx, cy) in enumerate(((120, 205), (400, 205), (120, 435), (400, 435), (200, 320), (320, 320))):
        ring = np.abs(np.hypot(xx - cx, yy - cy) - 16) < 3
        img[ring] = _rgb((0.9, 0.9, 0.9))
        draw_text(img, str(i + 1), cx - 5, cy - 9, (0.95, 0.95, 0.95), 2)
    # right panel: camera tiles
    for k, (title, y0) in enumerate((("EYES CAMERA", 96), ("WRIST CAMERA", 316))):
        img[y0:y0 + 204, 524:1000] = _rgb((0.11, 0.14, 0.18))
        draw_text(img, title, 540, y0 + 12, (0.7, 0.8, 0.9), 3)
        tile = img[y0 + 44:y0 + 192, 540:984]
        g = np.linspace(0.18, 0.30, tile.shape[0])[:, None, None]
        tile[:] = g * _rgb((0.8, 0.9, 1.0))
    img[536:560, 24:40] = _rgb((0.2, 0.8, 0.3))
    draw_text(img, "MODE  WORK", 52, 536, (0.85, 0.95, 0.85), 3)
    return to_uint8(img)
