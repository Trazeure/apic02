FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app
RUN pip install --no-cache-dir -r requirements.txt
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-80}