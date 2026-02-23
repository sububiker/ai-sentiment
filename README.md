# AI Sentiment App on Docker Desktop Kubernetes

## Run locally (no k8s)
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 8080

## Build Docker image
docker build -t ai-sentiment:local .

## Deploy to Docker Desktop Kubernetes
kubectl config use-context docker-desktop
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

### Using Helm
A Helm chart is provided under `charts/ai-sentiment` so you can install the application with template parameters:

```bash
# install the chart (namespace optional)
helm install ai-sentiment charts/ai-sentiment \
  --set image.repository=ai-sentiment \
  --set image.tag=local \
  --set service.nodePort=30080

# to upgrade after changing values
helm upgrade ai-sentiment charts/ai-sentiment

# uninstall
helm uninstall ai-sentiment
```

The original manifest files have been archived under `k8s/archive/` for reference.

## Test
curl http://localhost:30080/health
curl http://localhost:30080/predict -H "Content-Type: application/json" -d '{"text":"i love this"}'# ai-sentiment
