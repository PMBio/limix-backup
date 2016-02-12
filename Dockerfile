# Limix 

# base image from python 2.7 with miniconda
FROM continuumio/miniconda

ENV PYTHONUNBUFFERED 1

RUN mkdir /code

WORKDIR /code

ADD . /code

RUN conda env create
RUN source activate limix

# Adding the `code` directory to the path, so we can execute the script.
ENV PATH /code:$PATH
