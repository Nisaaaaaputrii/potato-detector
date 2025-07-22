FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install ngrok
RUN wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz \
    && tar xvzf ngrok-v3-stable-linux-arm64.tgz -C /usr/local/bin \
    && rm ngrok-v3-stable-linux-arm64.tgz

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Authenticate ngrok if token is provided\n\
if [ ! -z "$NGROK_AUTHTOKEN" ]; then\n\
    ngrok config add-authtoken $NGROK_AUTHTOKEN\n\
fi\n\
\n\
# Start Streamlit in background\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &\n\
\n\
# Wait a moment for Streamlit to start\n\
sleep 5\n\
\n\
# Start ngrok tunnel\n\
ngrok http 8501 --log=stdout\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port for internal use only (not published)
EXPOSE 8501

# Start the application
CMD ["/app/start.sh"]