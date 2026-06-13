"""Export the reviewed shortlist as report-ready three-column galleries."""

from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import FileLink, display
from PIL import Image


if "selected" not in globals() or len(selected) < 10:
    raise RuntimeError("Chay Cell 24 truoc va dam bao co du 10 cap selected.")

TOP_ROWS = selected[:10]
ROWS_PER_IMAGE = 5
PANEL_COUNT = 5
PANEL_GAP = 5
HEADER_HEIGHT = 30


def _crop_comparison_panels(comparison_path):
    image = Image.open(comparison_path).convert("RGB")
    width, height = image.size
    panel_width = (width - PANEL_GAP * (PANEL_COUNT - 1)) // PANEL_COUNT

    def crop_panel(index):
        left = index * (panel_width + PANEL_GAP)
        return image.crop((left, HEADER_HEIGHT, left + panel_width, height))

    return crop_panel(0), crop_panel(1), crop_panel(4)


saved_paths = []
for part_index in range(2):
    page_rows = TOP_ROWS[
        part_index * ROWS_PER_IMAGE:(part_index + 1) * ROWS_PER_IMAGE
    ]

    fig, axes = plt.subplots(
        len(page_rows),
        3,
        figsize=(10, 4.4 * len(page_rows)),
        squeeze=False,
    )

    headers = ["Person", "Target cloth", "Final result"]
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=12, fontweight="bold", pad=14)

    for row_index, row in enumerate(page_rows):
        panels = _crop_comparison_panels(row["comparison_file"])
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(panel)
            axes[row_index, column].axis("off")

        display_index = part_index * ROWS_PER_IMAGE + row_index + 1
        axes[row_index, 0].set_ylabel(
            f'{display_index:02d}\nperson={row["person_id"]}\ncloth={row["cloth_id"]}',
            fontsize=9,
            rotation=0,
            labelpad=65,
            va="center",
        )

    fig.suptitle(
        f"VTO V2 Epoch 12 - Selected Unpaired Results ({part_index + 1}/2)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.94, hspace=0.22, wspace=0.08)

    save_path = Path(
        f"/kaggle/working/vto_v2_top10_person_cloth_result_part_{part_index + 1}.jpg"
    )
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    saved_paths.append(save_path)

for save_path in saved_paths:
    print(f"Saved: {save_path}")
    display(FileLink(str(save_path)))
