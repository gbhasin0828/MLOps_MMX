# Use Python as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask is running on
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]
