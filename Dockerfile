FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app
COPY ./start.sh /code/start.sh

RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000 (Railway asignará el puerto real)
EXPOSE 8000

# Establecer permisos de ejecución para el script
RUN chmod +x /code/start.sh

# Usar el script de inicio
CMD ["/code/start.sh"]
