FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app

COPY requirements.lock .
RUN pip install --requirement requirements.lock

COPY schedule_risk_agent ./schedule_risk_agent
COPY schedule_risk_feature_calculation.sql schedule_risk_feature_store_refresh_current.sql schedule_risk_label_calculation.sql ./
COPY models ./models

RUN mkdir -p /var/lib/schedule-risk/features && \
    chown -R app:app /var/lib/schedule-risk /app
USER app

EXPOSE 8011
CMD ["python", "-m", "schedule_risk_agent.server"]

