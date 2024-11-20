FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app
RUN pip install --no-cache-dir -r requirements.txt

# Script para manejar el puerto
COPY ./start.sh /code/start.sh
RUN chmod +x /code/start.sh
CMD ["/code/start.sh"]