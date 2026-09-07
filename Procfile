web: python manage.py migrate && python manage.py collectstatic --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --max-requests 100 --max-requests-jitter 10

