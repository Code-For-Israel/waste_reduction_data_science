# set base image (host OS)
FROM pure/python:3.8-cuda10.2-base

# set the working directory in the container
WORKDIR /code

# copy the content of the local directory to the working directory
COPY env.yml .

# install dependencies
RUN rm /etc/apt/sources.list.d/cuda.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt update

RUN apt-get update
RUN apt-get -y install curl

# anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH="/opt/conda/bin:$PATH"
ENV PATH="/usr/local/cuda/bin:$PATH"
# pip and install env.yml to the base python environment
RUN pip install --upgrade pip
RUN conda init bash && conda env update --name base --file env.yml --prune
