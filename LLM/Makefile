default: build

help:
	@echo 'Management commands for kds_hack:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the airflow_pipeline project project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t kds_hack 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus all --ipc=host --name kds_hack -v `pwd`:/workspace kds_hack:latest /bin/bash

up: build run

rm: 
	@docker rm kds_hack

stop:
	@docker stop kds_hack

reset: stop rm