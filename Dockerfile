FROM python:3.11-slim

WORKDIR /app

COPY transformer_nowcasting.py .
COPY data_from_2024 ./data_from_2024

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "transformer_nowcasting.py"]