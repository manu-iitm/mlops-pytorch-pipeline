# MLOps PyTorch Pipeline

An end-to-end MLOps pipeline for training and serving an image classification model using PyTorch, Docker, and Kubernetes.

## Architecture

```text
                    +--------------------+
                    |   ConfigMap        |
                    | training_config    |
                    +---------+----------+
                              |
                              v
+----------------------------------------------------+
|                Kubernetes Cluster                  |
|                                                    |
|  +---------------------------+                     |
|  | Training Job              |                     |
|  |---------------------------|                     |
|  | PyTorch + ResNet18        |                     |
|  | CIFAR-10 Dataset          |                     |
|  | Saves model checkpoint    |                     |
|  +------------+--------------+                     |
|               |                                    |
|               v                                    |
|       +---------------+                            |
|       | checkpoints   | PVC                        |
|       +-------+-------+                            |
|               |                                    |
|               v                                    |
|  +---------------------------+                     |
|  | Serving Deployment        |                     |
|  |---------------------------|                     |
|  | FastAPI                   |                     |
|  | 2 Replicas                |                     |
|  | Loads saved checkpoint    |                     |
|  +------------+--------------+                     |
|               |                                    |
|               v                                    |
|       +---------------+                            |
|       | Service       |                            |
|       | ClusterIP     |                            |
|       +-------+-------+                            |
|               |                                    |
+---------------+------------------------------------+
                |
                v
        Client / curl
```

---

## Project Structure

```text
.
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── data-pvc.yaml
│   ├── checkpoints-pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   └── serving-service.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
└── README.md
```

---

## Local Setup

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements/train.txt
pip install -r requirements/serve.txt
```

---

## Model Training

```bash
python -m src.train
```

Training configuration is loaded from:

```text
configs/training_config.yaml
```

Example output:

```json
{"epoch":1,"train_loss":1.56,"val_loss":1.36}
{"event":"checkpoint_saved"}
```

Generated checkpoint:

```text
checkpoints/classifier_v1.pt
```

---

## Model Serving

Start FastAPI:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8080
```

Health Check:

```bash
curl http://localhost:8080/health
```

Prediction:

```bash
curl -X POST \
-F "file=@image.jpg" \
http://localhost:8080/predict
```

---

## Docker

### Build Training Image

```bash
docker build \
-f docker/Dockerfile.train \
-t mlops-train:v1 .
```

### Run Training

```bash
docker run --rm \
-v $(pwd)/data:/app/data \
-v $(pwd)/checkpoints:/app/checkpoints \
mlops-train:v1
```

### Build Serving Image

```bash
docker build \
-f docker/Dockerfile.serve \
-t mlops-serve:v1 .
```

### Run Serving

```bash
docker run --rm \
-p 8080:8080 \
-v $(pwd)/checkpoints:/app/checkpoints \
mlops-serve:v1
```

---

## Kubernetes Deployment

### Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### Create ConfigMap

```bash
kubectl apply -f k8s/configmap.yaml
```

### Create PVCs

```bash
kubectl apply -f k8s/data-pvc.yaml
kubectl apply -f k8s/checkpoints-pvc.yaml
```

### Start Training Job

```bash
kubectl apply -f k8s/training-job.yaml
```

Monitor:

```bash
kubectl get jobs -n ml-training
kubectl logs job/mlops-training-job -n ml-training
```

### Deploy Serving

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
```

Check status:

```bash
kubectl get deployments -n ml-training
kubectl get pods -n ml-training
kubectl get svc -n ml-training
```

---

## API Endpoints

### Health

```http
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict

```http
POST /predict
```

Form field:

```text
file=<image>
```

Response:

```json
{
  "airplane": 0.95,
  "bird": 0.02,
  "ship": 0.01
}
```

---

## Technologies Used

- PyTorch
- Torchvision
- FastAPI
- Docker
- Kubernetes
- CIFAR-10
- ResNet18
- YAML Configurations

---

## Future Enhancements

- CI/CD pipeline
- Model registry integration
- MLflow experiment tracking
- Horizontal Pod Autoscaling
- GPU-enabled training
