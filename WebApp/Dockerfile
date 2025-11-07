# This image will contain uv pre-installed 
FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app


COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3333

CMD ["python" , "server.py"]