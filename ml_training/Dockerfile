FROM python:3.11.6
RUN apt-get update && apt-get install -y libpq-dev gcc ffmpeg libsm6 libxext6 libkrb5-dev python3-dev
RUN pip install --upgrade pip
RUN pip install poetry==1.7.1

WORKDIR /ml_training

COPY poetry.lock pyproject.toml /ml_training/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . /ml_training