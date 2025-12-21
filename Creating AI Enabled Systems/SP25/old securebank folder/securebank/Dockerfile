# Use the official Ubuntu 20.04 base image
# FROM ubuntu:20.04

# Use a lightweight Python base image
FROM python:3.9-slim

# Update and install required packages
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory inside the container
WORKDIR /securebank

# Copy the application files into the container
COPY . /securebank

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Start the Flask application
CMD ["python3", "app/app.py"]
