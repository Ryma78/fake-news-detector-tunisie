FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000 8501
CMD ["echo", "Use docker-compose to run services"]