# Use the official Python image from Docker Hub
FROM python:3.12-slim

ENV PYTHONPATH="/src"

# Install build dependencies
RUN apt-get update \
    && apt-get install -y curl build-essential libpq-dev

# Install Poetry
ENV POETRY_VERSION=1.6.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:/root/.local/bin"

# Set the working directory
WORKDIR /src

# Copy only the Poetry files first
COPY pyproject.toml poetry.lock ./

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Install dependencies (including dev dependencies if needed)
RUN poetry install --no-interaction --no-ansi

# Copy the rest of your application code
COPY . .

# Copy .env file
COPY .env .

# Copy config file
COPY config.py .

# Expose the port your Flask app runs on (default is 5000)
EXPOSE 8000

# Set environment variables (if any)
ENV FLASK_APP=app.py

# Run the application
CMD ["python", "./src/api/app.py"]
