from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType

Image = t.Annotated[Path, ContentType("image/*")]

image = bentoml.images.Image(python_version='3.11', lock_python_packages=False) \
    .system_packages('libglib2.0-0t64', 'libsm6', 'libxext6', 'libxrender1', 'libgl1-mesa-dri') \
    .requirements_file('requirements.txt')

@bentoml.service(resources={"gpu": 1}, image=image)
class YoloService:
    def __init__(self):
        from ultralytics import YOLO

        yolo_model = os.getenv("YOLO_MODEL", "yolo11n.pt")

        self.model = YOLO(yolo_model)

    @bentoml.api(batchable=True)
    def predict(self, images: list[Image]) -> list[list[dict]]:
        results = self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api
    def render(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output
