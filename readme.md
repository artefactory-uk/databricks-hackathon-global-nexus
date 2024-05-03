# Global Nexus Hackathon
This repo contains our submission to the DataBricks Gen AI Hackathon.
The use case selected is Healthcare. We aim to create a more efficient way for researchers to quickly retrieve, understand and query research papers so they can do their literature review. We have used a RAG to provide context to the LLM

## When you first close this repo run the following commands to properly initialise your local development environment
Create you local virtual environment (change python version as required):
```
python3.11 -m venv _venv
```
Always activate and work only from the virtual environment:
```
source _venv/bin/activate
```
Install pre-commit hooks:
```
pre-commit install
```
## Default requirements
The requirements.txt file contains default requirements which are often relevant to data repository work.



## Running the Front End
To run the front end, run the following command from the root folder in the repository
```
python src/front_end/similarity_and_summarisation_service.py
```
