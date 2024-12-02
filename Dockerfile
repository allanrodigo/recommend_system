# Etapa 1: Construção do ambiente com as dependências
FROM python:3.12-slim AS builder

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PATH="$PATH:/root/.local/bin"

# Instalar dependências do sistema necessárias para construir dependências Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar o Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Definir diretório de trabalho
WORKDIR /app

# Copiar apenas os arquivos de dependências
COPY pyproject.toml poetry.lock ./

# Instalar as dependências (sem dependências de desenvolvimento)
RUN poetry install --no-dev

# Etapa 2: Construir a imagem final
FROM python:3.12-slim

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PATH="$PATH:/root/.local/bin"

# Instalar dependências de runtime necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar as dependências instaladas da etapa de build
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar o código da aplicação
COPY . .

# Expor a porta
EXPOSE 8000

# Set environment variables (if any)
ENV FLASK_APP=app.py

# Run the application
CMD ["python", "./src/api/app.py"]
