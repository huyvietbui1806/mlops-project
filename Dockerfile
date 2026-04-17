FROM python:3.11-slim

# Copy uv binary từ image chính thức của uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Thư mục làm việc trong container
WORKDIR /app

# Copy file dependency trước để tận dụng cache Docker
COPY pyproject.toml uv.lock ./

# Cài dependencies vào .venv trong /app
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# Copy source code sau
COPY . .

# Expose cổng FastAPI
EXPOSE 8000

# Chạy app
CMD ["/app/.venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

