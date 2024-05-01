# OSS-LLM
Build a RAG application using Llama 3 70B, Langchain and Groq
In this example we will load the contents from couple of langchain blog posts. 
We use Langchain's Webbaseloader for this exercise

## Create a virtual environment,it saves lot of headaches later

```bash
conda create -name <nameofenvironment> <specify python version>
```
For example:
conda create --name myenv python=3.10.0

## Activate the virtual environment
```bash
conda activate myenv
```
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

```bash
pip install -r requirements.txt
```
## Run the application
```bash
python -m main.py
```
## License

[MIT](https://choosealicense.com/licenses/mit/)
