# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD ./src /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable (just an arbitrary example to show how!)
ENV NAME World

# Run app.py when the container launches
CMD ["python", "RedWineQualityFlaskRun.py"]