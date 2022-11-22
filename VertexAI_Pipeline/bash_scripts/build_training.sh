docker system prune
docker build -t model-training -f ./components/training/Dockerfile .
docker tag model-training us-central1-docker.pkg.dev/arboreal-path-365814/vertexai-training/model-training-mlops:latest
docker push us-central1-docker.pkg.dev/arboreal-path-365814/vertexai-training/model-training-mlops:latest
