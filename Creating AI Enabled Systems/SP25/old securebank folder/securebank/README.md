# SecureBank Fraud Detection API

## Overview
This Dockerized Flask application provides an API endpoint to predict whether a given transaction is fraudulent or legitimate. It utilizes a dummy machine learning model that returns a random prediction.

## Prerequisites
- Install [Docker](https://docs.docker.com/get-docker/)
- (Optional) Install `curl` for testing the API

## Building and Running the Docker Container

1. **Clone this repository**:
   ```sh
   git clone <repo-url>
   cd securebank

2. docker build -t securebank-api .

3. docker run -p 5000:5000 securebank-api

4. Navigate to the securebank directory then test the Flask endpoint:

curl -X POST "http://localhost:5000/predict" -H "Content-Type: application/json" -d @test.json

5. docker ps  # Find the container ID
docker stop <container_id>

6. docker rm <container_id>

7. docker rmi securebank-api

