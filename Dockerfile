FROM ubuntu:latest
MAINTAINER fnndsc "vuhuutiep@gmail.com"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install -y git

RUN pip3 install git+https://github.com/aivivn/d2l-book

RUN mkdir d2l
WORKDIR /d2l

# CMD ["d2lbook", "build", "html"]
