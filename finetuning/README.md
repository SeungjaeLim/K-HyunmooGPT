# Finetuning

## Project Overview
The Finetuning project is designed to fine-tune a causal language model for question-answering tasks, leveraging the Hugging Face `transformers` library. It integrates Docker for consistent environment management and utilizes advanced techniques like model quantization and Parameter-Efficient Fine-tuning (PEFT).

## Key Features
- Fine-tuning a pre-trained causal language model.
- Custom data preprocessing for question-answering format.
- Docker integration for consistent and isolated runtime environments.
- Utilization of advanced techniques like model quantization and PEFT.

## Prerequisites
- Docker
- Python 3.6 or later (for running scripts outside Docker)

## Setup and Execution

### Docker Environment Setup
1. **Build Docker Image**: 
   - Run `make build` to create a Docker image based on the provided `Dockerfile`.
2. **Start Docker Container**:
   - Execute `make run` to start the Docker container with GPU support and appropriate configurations.

### Running the Python Script (finetuning.py)
1. **Script Overview**: `finetuning.py` handles the entire process of data loading, model initialization, training, and text generation.
2. **Execution**: Inside the Docker container, navigate to the script's directory and run `python finetuning.py`.

### Makefile Commands
- `make help`: Lists available management commands.
- `make preprocess`: Command for data preprocessing (requires completion in the script).
- `make up`: Builds and runs Docker container sequentially.
- `make stop`: Stops the Docker container.
- `make rm`: Removes the Docker container.
- `make reset`: Stops and removes the Docker container.

## Script Details (finetuning.py)
1. **Data Preprocessing**: Loads a CSV dataset and formats it for question-answering.
2. **Model and Tokenizer Initialization**: Sets up a causal language model with a custom BitsAndBytes configuration for efficiency.
3. **Training Setup**: Configures training parameters and prepares the model for fine-tuning.
4. **Text Generation**: Generates responses to test prompts, demonstrating the model's capability.

## Dockerfile and Requirements
- The `Dockerfile` sets up a PyTorch environment with CUDA support.
- `requirements.txt` lists all necessary Python libraries, ensuring consistency across environments.

## Project Folder Structure
- `finetuning.py`: Main Python script for model training and evaluation.
- `Makefile`: Simplifies Docker commands and project management tasks.
- `Dockerfile`: Defines the Docker environment.
- `requirements.txt`: Specifies Python dependencies.

## Notes
- Ensure the CSV dataset and model/tokenizer paths are correctly set in `finetuning.py`.
- Adjust training parameters and Docker configurations as per your hardware capabilities and requirements.
- The project is optimized for GPU usage; ensure Docker supports GPU in your setup.
