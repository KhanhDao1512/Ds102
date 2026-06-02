import numpy as np
from PIL import Image, ImageDraw


def draw_scatter(x, labels, centers, path, title):
    colors = [
        (34, 113, 179),
        (215, 89, 67),
        (64, 157, 93),
        (144, 99, 178),
        (219, 171, 62),
    ]
    width, height, margin = 760, 560, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)

    def project(points):
        p = (points - mins) / span
        px = margin + p[:, 0] * (width - 2 * margin)
        py = height - margin - p[:, 1] * (height - 2 * margin)
        return np.column_stack([px, py])

    draw.rectangle((margin, margin, width - margin, height - margin), outline=(180, 180, 180))
    draw.text((margin, 20), title, fill=(0, 0, 0))

    pix = project(x)
    for point, label in zip(pix, labels):
        px, py = point
        color = colors[int(label) % len(colors)]
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)

    center_pix = project(centers)
    for center in center_pix:
        cx, cy = center
        draw.line((cx - 8, cy, cx + 8, cy), fill=(0, 0, 0), width=3)
        draw.line((cx, cy - 8, cx, cy + 8), fill=(0, 0, 0), width=3)

    image.save(path)

