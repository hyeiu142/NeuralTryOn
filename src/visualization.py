"""Visualization helpers shared by evaluation notebooks."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def save_three_column_comparison(
    person: Image.Image,
    target_cloth: Image.Image,
    result: Image.Image,
    output_path: str | Path,
    sample_label: str = "",
    size: tuple[int, int] = (384, 512),
) -> None:
    """Save a Person | Target Cloth | Try-On Result comparison image."""
    width, height = size
    header_height = 42
    gap = 8
    font = ImageFont.load_default()
    panels = [
        ("Person Input", person),
        ("Target Cloth", target_cloth),
        ("Try-On Result", result),
    ]

    canvas = Image.new(
        "RGB",
        (3 * width + 2 * gap, height + header_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = index * (width + gap)
        canvas.paste(panel.convert("RGB").resize(size, Image.BILINEAR), (x, header_height))
        draw.text((x + 8, 8), title, fill="black", font=font)
    if sample_label:
        draw.text((8, header_height - 15), sample_label, fill="black", font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
