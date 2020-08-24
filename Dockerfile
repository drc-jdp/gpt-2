FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

ARG DEFAULT_RESTORE_FROM

ENV RESTORE_FROM=${DEFAULT_RESTORE_FROM}

# RUN apt-get update 
# RUN apt-get install -y vim
# RUN apt-get install -y sudo

RUN mkdir /home/gpt-training
WORKDIR /home/gpt-training 

# boot up and setting files
COPY boot.sh /bin
COPY .setenv /
RUN cat /.setenv > /etc/bash.bashrc

COPY *.py ./
COPY requirements.txt ./

RUN mkdir src
COPY src/*.py ./src/

RUN mkdir test
COPY test/*   ./test/

RUN mkdir dataset
COPY dataset/* ./dataset/

RUN mkdir -p models/ci_training
COPY models/* ./models/ci_training/

RUN pip install -r requirements.txt

CMD /bin/bash /bin/boot.sh ${RESTORE_FROM:-no}