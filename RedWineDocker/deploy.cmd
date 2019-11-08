docker build -t redwine .
docker run -dit -p 8080:8080 --name redwine redwine

REM docker exec -it redwine /bin/bash
REM docker logs -f CID


REM Save Docker Image to Azure Container Registry to use in things like Azure Kubernetes Services or Azure Container Instance 
REM 
REM az login
REM az acr login --name jbj2CR
REM docker tag redwine jbj2cr.azurecr.io/redwine:v1
REM docker push jbj2cr.azurecr.io/redwine:v1
REM az acr repository list --name jbjCR --output table