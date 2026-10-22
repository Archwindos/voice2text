"""生成 assets/voxscribe.ico（蓝色圆形麦克风图标，与托盘图标同风格）。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

SIZE = 256
image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

# 蓝色圆底
margin = 6
draw.ellipse((margin, margin, SIZE - margin, SIZE - margin), fill=(37, 99, 235, 255))

# 白色麦克风（按 64px 设计等比放大 4 倍）
scale = 4
ox = oy = 0


def rect(x1, y1, x2, y2, radius, color=(255, 255, 255, 255)):
    draw.rounded_rectangle(
        (ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale),
        radius=radius * scale,
        fill=color,
    )


def line(x1, y1, x2, y2, width, color=(255, 255, 255, 255)):
    draw.line(
        (ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale),
        fill=color,
        width=width * scale,
    )


def arc(x1, y1, x2, y2, start, end, width, color=(255, 255, 255, 255)):
    draw.arc(
        (ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale),
        start=start,
        end=end,
        fill=color,
        width=width * scale,
    )


# 64px 设计坐标（与 tray.create_tray_image 一致）
rect(26, 13, 38, 34, 6)
arc(20, 23, 44, 46, 0, 180, 3)
line(32, 46, 32, 51, 3)
line(24, 52, 40, 52, 3)

icon_path = ASSETS / "voxscribe.ico"
image.save(
    icon_path,
    format="ICO",
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
png_path = ASSETS / "voxscribe.png"
image.save(png_path, format="PNG")
print("saved:", icon_path, "size:", icon_path.stat().st_size, "bytes")
print("saved:", png_path, "size:", png_path.stat().st_size, "bytes")
