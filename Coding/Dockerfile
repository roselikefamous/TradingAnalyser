# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone or copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
# If you don't have a requirements.txt, you can install them directly
RUN pip3 install --no-cache-dir \
    streamlit \
    yfinance \
    pandas \
    numpy \
    scipy \
    plotly \
    alpaca-trade-api \
    streamlit-autorefresh \
    pydantic \
    python-dotenv \
    numba

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
