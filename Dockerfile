FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed, then Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY loan_approval_dataset.csv .
COPY train.py .
COPY app.py .

# Train the model during the image build step
RUN python train.py

# Streamlit network configuration
EXPOSE 8501

# Command to launch Streamlit in production mode
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

