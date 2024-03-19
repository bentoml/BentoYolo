from __future__ import annotations

import json
from pathlib import Path

import bentoml

YOLO_MODEL = "yolov8n.pt"


@bentoml.service(resources={"gpu": 1})
class YoloService:
    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(YOLO_MODEL)

    @bentoml.api(batchable=True)
    def predict(self, images: list[Path]) -> list[str]:
        results = self.model.predict(source=images)
        return [result.tojson() for result in results]


@bentoml.service
class YoloV8:
    yolo = bentoml.depends(YoloService)

    @bentoml.api
    def predict(self, image: Path) -> list[dict]:
        result = self.yolo.predict([image])[0]
        return json.loads(result)
