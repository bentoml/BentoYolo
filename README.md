<div align="center">
    <h1 align="center">Serving YOLO with BentoML</h1>
</div>

YOLO (You Only Look Once) is a series of popular convolutional neural network (CNN) models used for object detection tasks.

This is a BentoML example project, demonstrating how to build an object detection inference API server, using the [YOLOv8 model](https://huggingface.co/Ultralytics/YOLOv8). See [here](https://github.com/bentoml/BentoML/tree/main/examples) for a full list of BentoML example projects.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoYolo.git
cd BentoYolo

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-03-19T10:02:15+0000 [WARNING] [cli] Converting 'YoloV8' to lowercase: 'yolov8'.
2024-03-19T10:02:16+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:YoloV8" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@demo-image.jpg;type=image/jpeg'
```

Python client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(
        image=Path("demo-image.jpg"),
    )
```

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
