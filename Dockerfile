FROM debian:bookworm

RUN apt update
RUN apt install python3 wget unzip make gcc g++ build-essential clang pip -y

COPY . .

WORKDIR /usr/src/

RUN wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
RUN unzip v0.9.2.zip
WORKDIR /usr/src/fastText-0.9.2
RUN make


CMD [ "fasttext" ]
