FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app

RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto para Railway
EXPOSE 8000

# Usar un valor por defecto para PORT
ENV PORT=8000

# Ejecutar uvicorn asegurando que la variable PORT es interpretada correctamente
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
