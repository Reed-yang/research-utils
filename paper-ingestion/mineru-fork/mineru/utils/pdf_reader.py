# Copyright (c) Opendatalab. All rights reserved.
import base64
import os
from io import BytesIO

from loguru import logger
from PIL import Image
from pypdfium2 import PdfBitmap, PdfDocument, PdfPage


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def normalize_image_format(image_format: str | None) -> str:
    if not image_format:
        image_format = os.getenv("MINERU_IMAGE_FORMAT", "webp")
    normalized = image_format.strip().lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in ("jpeg", "png", "webp"):
        normalized = "webp"
    return normalized


def get_image_extension(image_format: str) -> str:
    if image_format == "jpeg":
        return ".jpg"
    if image_format == "png":
        return ".png"
    if image_format == "webp":
        return ".webp"
    return ".jpg"


def get_image_save_settings(
    image_format: str | None = None,
    quality: int | None = None,
    lossless: bool | None = None,
) -> tuple[str, int, bool]:
    fmt = normalize_image_format(image_format)
    if quality is None:
        quality = _env_int("MINERU_IMAGE_QUALITY", 95)
    quality = max(1, min(100, int(quality)))
    if lossless is None:
        lossless = _env_bool("MINERU_IMAGE_LOSSLESS", False)
    return fmt, quality, lossless


def resolve_render_dpi(dpi: int) -> int:
    return _env_int("MINERU_PDF_RENDER_DPI", dpi)


def resolve_render_max_side(max_width_or_height: int) -> int:
    return _env_int("MINERU_PDF_RENDER_MAX_SIDE", max_width_or_height)


def page_to_image(
    page: PdfPage,
    dpi: int = 300,
    max_width_or_height: int = 4500,  # increased default for sharper rendering
) -> (Image.Image, float):
    dpi = resolve_render_dpi(dpi)
    max_width_or_height = resolve_render_max_side(max_width_or_height)
    scale = dpi / 72

    long_side_length = max(*page.get_size())
    if (long_side_length*scale) > max_width_or_height:
        scale = max_width_or_height / long_side_length

    bitmap: PdfBitmap = page.render(scale=scale)  # type: ignore

    image = bitmap.to_pil()
    try:
        bitmap.close()
    except Exception as e:
        logger.error(f"Failed to close bitmap: {e}")
    return image, scale




def image_to_bytes(
    image: Image.Image,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str | None = None,
    quality: int | None = None,
    lossless: bool | None = None,
) -> bytes:
    fmt, quality, lossless = get_image_save_settings(image_format, quality, lossless)
    pil_format = "JPEG" if fmt == "jpeg" else fmt.upper()
    save_kwargs = {}
    if fmt == "jpeg":
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        save_kwargs.update(
            {"quality": quality, "optimize": True, "progressive": True}
        )
    elif fmt == "webp":
        save_kwargs.update(
            {
                "quality": 100 if lossless else quality,
                "lossless": lossless,
                "method": 6,
            }
        )
    elif fmt == "png":
        save_kwargs.update({"optimize": True})
    with BytesIO() as image_buffer:
        image.save(image_buffer, format=pil_format, **save_kwargs)
        return image_buffer.getvalue()


def image_to_b64str(
    image: Image.Image,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str | None = None,
    quality: int | None = None,
    lossless: bool | None = None,
) -> str:
    image_bytes = image_to_bytes(image, image_format, quality, lossless)
    return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_pil_image(
    base64_str: str,
) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_str)
    with BytesIO(image_bytes) as image_buffer:
        return Image.open(image_buffer).convert("RGB")


def pdf_to_images(
    pdf: str | bytes | PdfDocument,
    dpi: int = 300,
    max_width_or_height: int = 4500,
    start_page_id: int = 0,
    end_page_id: int | None = None,
) -> list[Image.Image]:
    doc = pdf if isinstance(pdf, PdfDocument) else PdfDocument(pdf)
    page_num = len(doc)

    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else page_num - 1
    if end_page_id > page_num - 1:
        logger.warning("end_page_id is out of range, use images length")
        end_page_id = page_num - 1

    images = []
    try:
        for i in range(start_page_id, end_page_id + 1):
            image, _ = page_to_image(doc[i], dpi, max_width_or_height)
            images.append(image)
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return images


def pdf_to_images_bytes(
    pdf: str | bytes | PdfDocument,
    dpi: int = 300,
    max_width_or_height: int = 4500,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str | None = None,
    quality: int | None = None,
    lossless: bool | None = None,
) -> list[bytes]:
    images = pdf_to_images(pdf, dpi, max_width_or_height, start_page_id, end_page_id)
    return [
        image_to_bytes(image, image_format, quality, lossless) for image in images
    ]


def pdf_to_images_b64strs(
    pdf: str | bytes | PdfDocument,
    dpi: int = 300,
    max_width_or_height: int = 4500,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    # image_format: str = "PNG",  # 也可以用 "JPEG"
    image_format: str | None = None,
    quality: int | None = None,
    lossless: bool | None = None,
) -> list[str]:
    images = pdf_to_images(pdf, dpi, max_width_or_height, start_page_id, end_page_id)
    return [
        image_to_b64str(image, image_format, quality, lossless) for image in images
    ]
