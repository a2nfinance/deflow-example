# Use the official Python image
FROM python:3.12.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script and any other necessary files/directories into the container
#COPY . .

# Specify the command to run your script
#CMD ["python", "your_script_name.py"]
