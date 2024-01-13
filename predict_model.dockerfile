# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_classifier/ mnist_classifier/
COPY data/ data/
COPY models/ models/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mnist_classifier/predict_model.py", "-model_path=models/2024-01-12_21-47-28/model.pth", "-data_path=data/processed/test_dataset.pt"]