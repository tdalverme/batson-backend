FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN apt-get update ##[edited]

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -U pip wheel cmake

RUN pip install -r requirements.txt

COPY src/ .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]