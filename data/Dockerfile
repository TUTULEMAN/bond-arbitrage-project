# Use official Python runtime as base image
FROM python:3.13-alpine

# Set working directory in container
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code
COPY src/ .

# Command to run the script
CMD ["python", "main.py"]