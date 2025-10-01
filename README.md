# Python Misinformation Detection AI

This repository contains a bunch of stuff that might resemble an artificial intelligence. 

## Cloning And Executing

### Using Nix

This project has a convenient nix shell environment you can use as an alternative to a python virtual environment. Download the package manager via NixOS or the [Nix package manager](https://nixos.org/download/). 

To switch environments, execute:

```
nix-shell
```

You are now inside the environment defined by shell.nix

### Using Python Virtual Environment

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

python preprocess-vicuni.py # Process Victoria University's datasets
python charts.py            # Display all data visualisations 
python import.py            # Import all data into ChromaDB
python query.py "<prompt>"  # Search data within the ChromaDB database
```

### Building `fasttext` on Podman

This should also work on Docker by the way.

To build the image, execute the following command:

```
podman image build . --tag "gomisinfoai"
```

Once that is done, you may then access the container using the following command:

```sh
podman run -it gomisinfoai fasttext <arguments>
```

## Usage 

### Using `preprocess-vicuni.py`

```
usage: Misinformation Dataset Preprocessor (Victoria University Dataset) [-h] [-t INPUT_TRUE] [-f INPUT_FAKE] [-o OUTPUT_TRAINING] [-l OUTPUT_TESTING] [-m IN_MEMORY] [-p TRAINING_PERCENT] [-y AUTO_ACCEPT] [-w HEDGE_WORD_FILE]

Preprocesses (cleans) everything from the victoria university dataset into an output csv file

options:
  -h, --help            show this help message and exit
  -t, --input-true INPUT_TRUE
                        The True.csv file from the Victoria University misinformation dataset. Default='./assets/vicuni/True.csv'
  -f, --input-fake INPUT_FAKE
                        The Fake.csv file from the Victoria University misinformation dataset. Default='./assets/vicuni/Fake.csv'
  -o, --output-training OUTPUT_TRAINING
                        The file to output the processed information. Default='./assets/preprocessed-training.csv'
  -l, --output-testing OUTPUT_TESTING
                        The file to output data used for testing. Default='./assets/preprocessed-testing.csv'
  -m, --in-memory IN_MEMORY
                        Whether the CSV processing should be done in-memory. You will need a lot of RAM on your system for this to work, but will have performance improvements. Default=False
  -p, --training-percent TRAINING_PERCENT
                        Percentage of data to use for training. Default=0.9
  -y, --auto-accept AUTO_ACCEPT
                        Whether the program should automatically accept inputs (such as overwriting files). Default=False
  -w, --hedge-word-file HEDGE_WORD_FILE
                        Path to the file containing comma-delimited hedge words. Default='./assets/hedge-words.txt'
```

### Using `import.py`

```
usage: Misinformation Dataset Importer [-h] [-i INPUT] [-o OLLAMA_URL] [-m OLLAMA_MODEL_NAME] [-c CHROMADB_PATH] [-d CHROMADB_COLLECTION_NAME]

Imports data from the preprocessor into a Chroma database for analysis and machine learning

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     The input CSV file from the preprocessor program. Default='./assets/preprocessed-training.csv'
  -o, --olama-url OLLAMA_URL
                        The URL to your running ollama instance. Default='http://localhost:11434'
  -m, --olama-model-name OLLAMA_MODEL_NAME
                        The model name for your ollama instance for embedding generation. Default='embeddinggemma:latest'
  -c, --chromadb-path CHROMADB_PATH
                        Path to the ChromaDB database. Default='/var/lib/chroma'
  -d, --chromadb-collection-name CHROMADB_COLLECTION_NAME
                        Path to the ChromaDB database. Default='misinformation'

This program depends on ollama, so please have that installed. Install guide can be found here: https://docs.ollama.com/quickstart
```

### Using `test.py`

```
usage: Misinformation Dataset Testeng [-h] [-i INPUT] [-c CHROMADB_PATH] [-d CHROMADB_COLLECTION_NAME] [-o OLLAMA_URL]
                                      [-e OLLAMA_EMBEDDING_MODEL] [-v VERBOSE_LEVEL] [-m MAX_TESTS]

Uses the test database to determine how effective the AI model is at it's job

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     The testing CSV file from the preprocessor program. Default='./assets/preprocessed-testing.csv'
  -c, --chromadb-path CHROMADB_PATH
                        Path to the ChromaDB database. Default='/var/lib/chroma'
  -d, --chromadb-collection-name CHROMADB_COLLECTION_NAME
                        Path to the ChromaDB database. Default='misinformation'
  -o, --olama-url OLLAMA_URL
                        URL to Ollama instance. Default='localhost:11434'
  -e, --olama-embedding-model OLLAMA_EMBEDDING_MODEL
                        The model to use to create embeddings for queries. Should be the same as what you used when
                        importing into chromadb. Default='embeddinggemma:latest'
  -v, --verbose-level VERBOSE_LEVEL
                        0=nothing (default), 1=show incorrect matches, 2=show incorrect matches and distances, 3=show
                        everything
  -m, --max-tests MAX_TESTS
                        amount of tests to do in total (used for debugging). 0 will process all records. Default=0
```
