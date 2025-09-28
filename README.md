# Python Misinformation Detection AI

This repository contains a bunch of stuff that might resemble an artificial intelligence. 

## Cloning And Executing

Because I have no idea how to use Python, I have added `assets/venv` into the `gitignore` file, so you may create & install dependencies there. 

### Setup

Firstly, you will need to install Ollama on your system. If your using Arch Linux, it is easier to install it manually (https://wiki.archlinux.org/title/Ollama) instead of through podman. The reason I prefer using ollama over the default embedding function is that it is easier to configure gpu acceleration, making this process much quicker.

Once that is done, enable the service and execute `ollama pull embeddinggemma:latest`. I hardcoded this value into the codebase, so you have no choice unless you change it yourself. This can be updated upon request, I just don't have enough time now.

To configure chroma and actually run the programs,

```sh
# I will make this optional in the future, but I like 
# having everything in /var/lib so whatever
sudo mkdir /var/lib/chroma/
sudo chown $(USER):$(USER) -R /var/lib/chroma/

cd assets
python -m venv ./venv
source venv/bin/activate
pip install pandas matplotlib chromadb rich ollama

python preprocess-vicuni.py # Process Victoria University's datasets
python charts.py            # Display all data visualisations 
python import.py            # Import all data into ChromaDB
python query.py "<prompt>"  # Search data within the ChromaDB database
```

### Using `preprocess-vicuni.py`

```
usage: Misinformation Dataset Preprocessor (Victoria University Dataset)
       [-h] [-t INPUT_TRUE] [-f INPUT_FAKE] [-o OUTPUT] [-m IN_MEMORY] [-y AUTO_ACCEPT]
       [-w HEDGE_WORD_FILE]

Preprocesses (cleans) everything from the victoria university dataset into an output csv
file

options:
  -h, --help            show this help message and exit
  -t, --input-true INPUT_TRUE
                        The True.csv file from the Victoria University misinformation
                        dataset. Default='./vicuni/True.csv'
  -f, --input-fake INPUT_FAKE
                        The Fake.csv file from the Victoria University misinformation
                        dataset. Default='./vicuni/Fake.csv'
  -o, --output OUTPUT   The file to output the processed information.
                        Default='./preprocessed.csv'
  -m, --in-memory IN_MEMORY
                        Whether the CSV processing should be done in-memory. You will need
                        a lot of RAM on your system for this to work, but will have
                        performance improvements. Default=False
  -y, --auto-accept AUTO_ACCEPT
                        Whether the program should automatically accept inputs (such as
                        overwriting files). Default=False
  -w, --hedge-word-file HEDGE_WORD_FILE
                        Path to the file containing comma-delimited hedge words.
                        Default='./hedge-words.txt'
```

### Using `import.py`

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
