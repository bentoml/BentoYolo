from __future__ import annotations

import json
import typing as t
from pathlib import Path

import bentoml
from bentoml.validators import ContentType

Image = t.Annotated[Path, ContentType("image/*")]

YOLO_MODEL = "yolov8n.pt"


@bentoml.service
class YoloV8:
    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(YOLO_MODEL)

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
