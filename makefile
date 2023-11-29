default: build

build:
	@echo "Building Docker image"
	@docker build . -t fast_chroma 

run:
	@echo "Booting up Docker Container"
	@docker run -it  -p 8000:8000 --gpus '"device=0"' --ipc=host --name fast_chroma -v ./app:/usr/src/app fast_chroma:latest

up: build run

rm: 
	@docker rm fast_chroma

stop:
	@docker stop fast_chroma

reset: stop rm