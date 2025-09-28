FROM python:3.14.0rc3-trixie

WORKDIR /usr/src/app

RUN apt update
RUN apt install bash -y

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "bash" ]
