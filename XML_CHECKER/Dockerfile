# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .

COPY torchrequirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r torchrequirements.txt --index-url https://download.pytorch.org/whl/cpu
    # pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8001

# Start the application
CMD ["python3", "XmlChecker.py"]
