import numpy as np
from PIL import Image

from lab5.core.gmm import gmm_em
from lab5.utils.paths import OUT, ROOT, ensure_output_dir


def build_pixel_features(image):
    pixels = np.asarray(image, dtype=float) / 255.0
    h, w, _ = pixels.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.column_stack(
        [
            pixels.reshape(-1, 3),
            (xx.reshape(-1) / max(w - 1, 1))[:, None],
            (yy.reshape(-1) / max(h - 1, 1))[:, None],
        ]
    )


def border_pixel_mask(width, height):
    border = np.zeros(width * height, dtype=bool)
    border[:width] = True
    border[-width:] = True
    border[::width] = True
    border[width - 1 :: width] = True
    return border


def run_gmm_background_filter(image_name="cow.jpg", k=3):
    ensure_output_dir()
    image = Image.open(ROOT / image_name).convert("RGB")
    w, h = image.size
    features = build_pixel_features(image)

    model = gmm_em(features, k=k, max_iter=50, seed=11, reg=1e-5)
    responsibilities = model["responsibilities"]

    border = border_pixel_mask(w, h)
    background_component = responsibilities[border].mean(axis=0).argmax()
    background_prob = responsibilities[:, background_component].reshape(h, w)
    foreground_mask = background_prob < 0.5

    original = np.asarray(image)
    white_bg = np.full_like(original, 255)
    filtered = np.where(foreground_mask[:, :, None], original, white_bg)
    mask_image = foreground_mask.astype(np.uint8) * 255

    Image.fromarray(filtered).save(OUT / "cow_foreground_gmm.png")
    Image.fromarray(mask_image).save(OUT / "cow_foreground_mask.png")

    rgba = np.dstack([original, mask_image])
    Image.fromarray(rgba).save(OUT / "cow_foreground_transparent.png")

    return {
        "image_size": (w, h),
        "background_component": int(background_component),
        "foreground_ratio": float(foreground_mask.mean()),
        "outputs": [
            "cow_foreground_gmm.png",
            "cow_foreground_mask.png",
            "cow_foreground_transparent.png",
        ],
    }

