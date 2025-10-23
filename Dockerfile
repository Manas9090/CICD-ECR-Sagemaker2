FROM python:3.11-slim

# Create directories
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/code

# Set working directory
WORKDIR /opt/ml/code

# Copy training and inference code
COPY train.py .
COPY inference.py .
COPY requirements.txt .
COPY model/ /opt/ml/model/

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for inference
EXPOSE 8080

# Start the inference service
CMD ["python", "inference.py"] 
