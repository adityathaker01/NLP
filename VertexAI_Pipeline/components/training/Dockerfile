FROM python:3.7
WORKDIR /pipeline

COPY ./components/training/ .

RUN pip install --upgrade pip

# COPY components/data_prep/requirements.txt .

RUN pip3 install -r requirements.txt
