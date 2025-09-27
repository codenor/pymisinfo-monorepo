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
- `rich`

Because I have no idea how to use Python, I have added `assets/venv` into the `gitignore` file, so you may create & install dependencies there. 

Why am I using Python for this? I thought I needed it, and it turned out that I didn't really need to in the first place.

### Setup

```sh
# I will make this optional in the future, but I like 
# having everything in /var/lib so whatever
sudo mkdir /var/lib/chroma/
sudo chown $(USER):$(USER) -R /var/lib/chroma/

python -m venv ./venv
pip install pandas matplotlib chromadb rich

cd assets
python charts.py           # Display all data visualisations 
python import.py           # Import all data into ChromaDB
python query.py "<prompt>" # Search data within the ChromaDB database
```
