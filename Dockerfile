FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py pytest.ini requirements.txt requirements-log.txt ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY scripts ./scripts
COPY Makefile README.md ./

RUN pip install --no-cache-dir -e . -r requirements.txt \
    && pip install --no-cache-dir ruff

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Default: quality gate (no GPU, no Waymo images).
CMD ["python", "-m", "pytest", "-q"]
