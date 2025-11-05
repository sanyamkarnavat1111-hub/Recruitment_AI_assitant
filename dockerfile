
FROM python:3.11-slim


# Copy only requirements first (better build caching)
COPY requirements.txt .


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
