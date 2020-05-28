FROM ubuntu:latest
MAINTAINER fnndsc "vuhuutiep@gmail.com"

RUN apt-get update \
  && apt-get -y install software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get install -y python3.7 \
  && apt-get install -y git \
  && apt-get install -y curl \
  && apt-get install -y zip

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN mkdir d2l
WORKDIR /d2l

# libgomp1 to fix this error https://github.com/explosion/spaCy/issues/1110
RUN apt-get install -y pandoc libgomp1

# clone and build d2l-book, changing D2L_VER will break the cache here
ARG D2L_VER=unknown
# Chỗ này mình để install d2l-book và d2l-en ở forked repos của mình,
# vì nếu để như cũ hình như GitHub action clone code từ branch master của avivn đang cần sửa. 
# Sau nếu ổn mọi người chỉnh lại nhé.
RUN pip3 install git+https://github.com/cuongvng/d2l-book
RUN pip3 install --no-cache-dir mxnet==1.6.0b20191122 git+https://github.com/cuongvng/d2l-en

CMD ["d2lbook", "build", "html"]

  # && apt-get install -y wget \
  # && apt-get install -y unzip \
  # && apt-get install -y fontconfig
