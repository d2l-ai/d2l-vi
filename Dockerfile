FROM ubuntu:latest
MAINTAINER fnndsc "vuhuutiep@gmail.com"

RUN apt-get update \
  && apt-get -y install software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get install -y python3.7 \
  && apt-get install -y git \
  && apt-get install -y curl \
  && apt-get install -y wget \
  && apt-get install -y unzip \
  && apt-get install -y fontconfig


RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN mkdir d2l
WORKDIR /d2l

RUN apt-get install -y pandoc

# clone and build d2l-book, changing D2L_VER will break the cache here
ARG D2L_VER=unknown
RUN pip3 install git+https://github.com/aivivn/d2l-book

# install fonts
RUN wget https://www.fontsquirrel.com/fonts/download/source-serif-pro
RUN unzip -o source-serif-pro -d ~/.fonts/
RUN rm source-serif-pro

RUN wget https://www.fontsquirrel.com/fonts/download/source-sans-pro
RUN unzip -o source-sans-pro -d ~/.fonts/
RUN rm source-sans-pro

RUN fc-cache -f -v

CMD ["d2lbook", "build", "pdf"]
