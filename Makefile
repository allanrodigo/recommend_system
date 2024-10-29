# Variables
IMAGE_NAME = recommend
CONTAINER_NAME = recommend_container
PORT = 8000
DOCKERFILE = Dockerfile
DOCKER_CONTEXT = .
ENV_FILE = .env

# Default target
.PHONY: help
help:
	@echo "Makefile for managing the Flask Docker application"
	@echo ""
	@echo "Available targets:"
	@echo "  build        Build the Docker image"
	@echo "  run          Run the Docker container"
	@echo "  stop         Stop the running container"
	@echo "  remove       Remove the container"
	@echo "  logs         View container logs"
	@echo "  shell        Open a shell inside the running container"
	@echo "  rebuild      Stop, remove, rebuild, and run the container"
	@echo "  clean        Remove the image and container"

# Build the Docker image
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):latest $(DOCKER_CONTEXT)

# Run the Docker container
.PHONY: run
run:
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME):latest

# Stop the running container
.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME)

# Remove the container
.PHONY: remove
remove:
	docker rm $(CONTAINER_NAME)

# View container logs
.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Open a shell inside the running container
.PHONY: shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Rebuild the image and run the container
.PHONY: rebuild
rebuild: stop remove build run

# Clean up the image and container
.PHONY: clean
clean: stop remove
	docker rmi $(IMAGE_NAME):latest

test:
	poetry run pytest
