docker system prune
docker build -t data-prep -f ./components/data_prep/Dockerfile .
docker tag data-prep us-central1-docker.pkg.dev/arboreal-path-365814/vertexai-training/data-prep-mlops:latest
docker push us-central1-docker.pkg.dev/arboreal-path-365814/vertexai-training/data-prep-mlops:latest