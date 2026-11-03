# ============================================================
# 14. IMAGE CONFIGURATION
# ============================================================

image_path = None

for filename in [
    "images.png",
    "images.jpeg",
    "images.jpg",
    "image.png",
]:
    candidate = APP_DIR / filename
    if candidate.is_file():
        image_path = candidate
        break
