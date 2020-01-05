FROM ubuntu:latest
FROM xucheng/texlive-full:latest
MAINTAINER fnndsc "vuhuutiep@gmail.com"

RUN apt-get update \
  && apt-get -y install software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get install -y python3.7 \
  && apt-get install -y git \
  && apt-get install -y curl

RUN apt-get install -y pandoc

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN mkdir d2l
WORKDIR /d2l


# clone and build d2l-book, changing D2L_VER will break the cache here
ARG D2L_VER=unknown
RUN pip3 install git+https://github.com/aivivn/d2l-book

CMD ["d2lbook", "build", "all"]
