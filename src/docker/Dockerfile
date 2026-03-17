FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN pip install uv && uv lock --upgrade-package gunicorn && uv sync --no-cache


COPY . .

CMD [ "uv", "run", "main.py" ]