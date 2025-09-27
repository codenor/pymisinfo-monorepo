# Go + Python Misinformation AI

This is a massive monorepo containing a bunch of things that might resemble an artificial intelligence. 

Golang is used to import and pre-process the data. Python is used as the actual artificial intelligence part of the project.

## Cloning And Executing

Ensure that when you clone this repo, you place it under `~/go/src/` or whatever you have configured that as. 

### Preprocessor

Running the preprocessor is as simple as running `go run .`

## Python 

- `pandas`
- `matplotlib`
- `chromadb`
- `ollama`
- `rich`

Because I have no idea how to use Python, I have added `assets/venv` into the `gitignore` file, so you may create & install dependencies there. 

Why am I using Python for this? I thought I needed it, and it turned out that I didn't really need to in the first place.

### Setup

Firstly, you will need to install Ollama on your system. If your using Arch Linux, it is easier to install it manually (https://wiki.archlinux.org/title/Ollama) instead of through podman. The reason I prefer using ollama over the default embedding function is that it is easier to configure gpu acceleration, making this process much quicker.

Once that is done, enable the service and execute `ollama pull embeddinggemma:latest`. I hardcoded this value into the codebase, so you have no choice unless you change it yourself. This can be updated upon request, I just don't have enough time now.

To configure chroma and actually run the programs,

```sh
# I will make this optional in the future, but I like 
# having everything in /var/lib so whatever
sudo mkdir /var/lib/chroma/
sudo chown $(USER):$(USER) -R /var/lib/chroma/

python -m venv ./venv
source venv/bin/activate
pip install pandas matplotlib chromadb rich ollama

cd assets
python charts.py           # Display all data visualisations 
python import.py           # Import all data into ChromaDB
python query.py "<prompt>" # Search data within the ChromaDB database
```

### `input.py`

```
usage: Misinformation Dataset Importer [-h] [-i INPUT] [-o OLLAMA_URL] [-m OLLAMA_MODEL_NAME] [-c CHROMADB_PATH]

Imports data from the preprocessor into a Chroma database for analysis and machine learning

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     The input CSV file from the preprocessor program
  -o, --olama-url OLLAMA_URL
                        The URL to your running ollama instance
  -m, --olama-model-name OLLAMA_MODEL_NAME
                        The model name for your ollama instance for embedding generation
  -c, --chromadb-path CHROMADB_PATH
                        Path to the ChromaDB database

This program depends on ollama, so please have that installed. Install guide can be found here: https://docs.ollama.com/quickstart
```
