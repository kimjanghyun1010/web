FROM python-nginx-uwsgi-django:latest
ENV PYTHONUNBUFFERED 1
COPY ./app /app
EXPOSE 80
CMD ["supervisord"]