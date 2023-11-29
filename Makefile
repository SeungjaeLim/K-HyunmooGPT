default: build

help:
	@echo 'Management commands for kds_hack3:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the airflow_pipeline project project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t kds_hack3

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus all --ipc=host --name kds_hack3 -v `pwd`:/workspace kds_hack2:latest /bin/bash

up: build run

rm: 
	@docker rm kds_hack3

stop:
	@docker stop kds_hack3

reset: stop rm