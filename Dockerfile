FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app

RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000 (Railway asignará el puerto real)
EXPOSE 8000

# Ejecutar uvicorn usando sh para interpretar correctamente $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
