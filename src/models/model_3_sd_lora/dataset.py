"""Model 3 dataset migrated from the completed Kaggle notebook."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, CLIPTokenizer

from .settings import Model3Settings


class Model3Dataset(Dataset):
    """Load paired identity reconstruction samples for SD + LoRA training."""

    def __init__(
        self,
        settings: Model3Settings,
        viton_root: str | Path,
        csv_root: str | Path,
        caption_root: str | Path,
        split: str = "train",
        augment: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.settings = settings
        self.split = split
        self.augment = augment
        self.root = Path(viton_root) / split
        self.csv_root = Path(csv_root)
        self.caption_dir = Path(caption_root) / "cloth-captions" / split
        self.height, self.width = settings.height, settings.width
        self.ratio = self.width / self.height

        if split == "train":
            frame = pd.read_csv(self.csv_root / "clean_vto_dataset_train.csv")
            self.pairs = [(str(value), str(value)) for value in frame["id"]]
        else:
            frame = pd.read_csv(self.csv_root / "clean_vto_dataset_test.csv")
            self.pairs = [(str(value), str(value)) for value in frame["person_id"]]
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

        self.tokenizer = CLIPTokenizer.from_pretrained(settings.model_id, subfolder="tokenizer")
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        person_id, cloth_id = self.pairs[index]
        person = Image.open(self.root / "image" / f"{person_id}.jpg").convert("RGB")
        agnostic = Image.open(self.root / "agnostic-v3.2" / f"{person_id}.jpg").convert("RGB")
        cloth = Image.open(self.root / "cloth" / f"{cloth_id}.jpg").convert("RGB")
        cloth_mask = Image.open(self.root / "cloth-mask" / f"{cloth_id}.jpg").convert("L")

        person = self._crop_resize(person)
        agnostic = self._crop_resize(agnostic)
        pose = self._crop_resize(self._load_pose(person_id))
        cloth = self._process_cloth(cloth, cloth_mask)
        mask = self._compute_parse_mask(person_id)

        if self.augment and random.random() < 0.5:
            person, agnostic, pose, mask = map(TF.hflip, (person, agnostic, pose, mask))

        caption = self._caption(cloth_id)
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "image": self._normalize(TF.to_tensor(person)),
            "agnostic": self._normalize(TF.to_tensor(agnostic)),
            "pose": self._normalize(TF.to_tensor(pose)),
            "cloth": self._normalize(TF.to_tensor(cloth)),
            "inpaint_mask": TF.to_tensor(mask),
            "clip_cloth": self.clip_processor(
                images=cloth, return_tensors="pt"
            ).pixel_values.squeeze(0),
            "input_ids": tokens.input_ids.squeeze(0),
            "caption": caption,
            "person_id": person_id,
            "cloth_id": cloth_id,
        }

    @staticmethod
    def _normalize(tensor):
        return tensor * 2.0 - 1.0

    def _crop_resize(self, image: Image.Image, is_mask: bool = False) -> Image.Image:
        width, height = image.size
        if width / height > self.ratio:
            new_width = int(height * self.ratio)
            image = image.crop(((width - new_width) // 2, 0, (width + new_width) // 2, height))
        else:
            new_height = int(width / self.ratio)
            image = image.crop((0, (height - new_height) // 2, width, (height + new_height) // 2))
        interpolation = Image.NEAREST if is_mask else Image.BILINEAR
        return image.resize((self.width, self.height), interpolation)

    def _find_file(self, folder: str, stem: str, extensions: tuple[str, ...]) -> Path | None:
        for extension in extensions:
            path = self.root / folder / f"{stem}{extension}"
            if path.exists():
                return path
        return None

    def _load_pose(self, person_id: str) -> Image.Image:
        path = self._find_file("image-densepose", person_id, (".jpg", ".png"))
        if path is None:
            path = self._find_file("openpose_img", f"{person_id}_rendered", (".png", ".jpg"))
        if path is None:
            return Image.fromarray(np.full((self.height, self.width, 3), 128, dtype=np.uint8))
        return Image.open(path).convert("RGB")

    def _compute_parse_mask(self, person_id: str) -> Image.Image:
        parse = self._find_file("image-parse-v3", person_id, (".png",))
        agnostic = self._find_file("image-parse-agnostic-v3.2", person_id, (".png",))
        if parse and agnostic:
            parse_array = np.array(Image.open(parse).resize((self.width, self.height), Image.NEAREST))
            agnostic_array = np.array(
                Image.open(agnostic).resize((self.width, self.height), Image.NEAREST)
            )
            mask = (parse_array != agnostic_array).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            return Image.fromarray(mask, mode="L")
        fallback = np.zeros((self.height, self.width), dtype=np.uint8)
        fallback[: int(self.height * 0.6), :] = 255
        return Image.fromarray(fallback, mode="L")

    def _process_cloth(self, cloth: Image.Image, mask: Image.Image) -> Image.Image:
        binary = Image.fromarray((np.array(mask) > 128).astype(np.uint8) * 255).convert("L")
        cutout = Image.composite(cloth, Image.new("RGB", cloth.size, "white"), binary)
        bbox = binary.getbbox()
        if bbox:
            cutout = cutout.crop(bbox)
        width, height = cutout.size
        pad_width, pad_height = max(1, int(width * 0.1)), max(1, int(height * 0.1))
        canvas = Image.new("RGB", (width + 2 * pad_width, height + 2 * pad_height), "white")
        canvas.paste(cutout, (pad_width, pad_height))
        canvas_width, canvas_height = canvas.size
        if canvas_width / canvas_height > self.ratio:
            final_height = int(canvas_width / self.ratio)
            fitted = Image.new("RGB", (canvas_width, final_height), "white")
            fitted.paste(canvas, (0, (final_height - canvas_height) // 2))
        else:
            final_width = int(canvas_height * self.ratio)
            fitted = Image.new("RGB", (final_width, canvas_height), "white")
            fitted.paste(canvas, ((final_width - canvas_width) // 2, 0))
        return fitted.resize((self.width, self.height), Image.BILINEAR)

    def _caption(self, cloth_id: str) -> str:
        path = self.caption_dir / f"{cloth_id}.txt"
        raw = path.read_text(encoding="utf-8").strip() if path.exists() else "a photo of a garment"
        return raw if self.settings.trigger_word in raw else f"{self.settings.trigger_word}, {raw}"

