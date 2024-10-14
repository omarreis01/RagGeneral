# Use an official Python runtime as a parent image
FROM python:3.12.3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Copy your environment variables file (base.env) into the container
COPY base.env /app/base.env

# Load environment variables from the env file
RUN export $(grep -v '^#' base.env | xargs)

# Install any necessary dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container (optional, based on your app requirements)
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the main application
CMD ["python", "main.py"]
