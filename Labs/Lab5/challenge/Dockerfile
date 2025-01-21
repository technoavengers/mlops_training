FROM apache/airflow:2.6.3

# Switch to root user to install packages
USER root

# Copy requirements.txt into the container
COPY requirements.txt /requirements.txt

# Switch to airflow user for pip installation
USER airflow

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
