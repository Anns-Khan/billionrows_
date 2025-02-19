# Use the official Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files into the container
COPY . /app

# Install pip and dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV WANDB_API_KEY="your_wandb_api_key"
ENV MODIN_ENGINE="dask"

# Expose the port that FastAPI runs on
EXPOSE 7860

# Start the Gradio app
CMD ["python", "-m", "gradio", "run", "your_script.py"]
