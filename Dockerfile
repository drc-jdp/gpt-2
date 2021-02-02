FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

# ARG DEFAULT_RESTORE_FROM
# ARG DEFAULT_LEARNING_RATE
# ARG DEFAULT_VAL_DATASET

# ENV RESTORE_FROM=${DEFAULT_RESTORE_FROM}
# ENV LEARNING_RATE=${DEFAULT_LEARNING_RATE}
# ENV VAL_DATASET=${DEFAULT_VAL_DATASET}

# boot up and setting files
COPY boot.sh /usr/bin
COPY .setenv /usr/bin
RUN cat /usr/bin/.setenv > /etc/bash.bashrc

RUN mkdir /home/gpt-training
WORKDIR /home/gpt-training 

COPY *.py ./
COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir src
COPY src/*.py ./src/

RUN mkdir test
COPY test/*   ./test/

RUN mkdir -p models/ci_training
COPY models/* ./models/ci_training/

RUN mkdir dataset

CMD /bin/bash /usr/bin/boot.sh ${RESTORE_FROM:-no} ${LEARNING_RATE:-0.00002} ${SAVE_EVERY:-1000}  \
    ${VAL_DATASET:-dataset} ${VAL_EVERY:-100} ${VAL_BATCH:-20}
