# Docker Usage
## model parameter
1.  --restore_from { `RF` } : 
    * latest  : train model with pre-trained model from saving place
      * need to have model in saving place
    * no  : pre-train the model only with the params in models/hparams.json and vocab
2.  --learning_rate { `LR` } : default: 0.00002
3.  --val_dataset { `VAL_D` } : the folder or file being the dataset for validation
4.  --save_every {`SAVE_EVERY` } : save the model for each `SAVE_EVERY` batch

## CI/CD
1. for **different model**, create new **branch** in GitHub
1. modify the hparams, vocab, encoder in {models/}
2. modify the model in train.py or src/model.py
3. push to GitHub, start CI/CD automatically
4. type on server
  * cpu
```bash
docker run -itd \
-e RESTORE_FROM={ RF } -e LEARNING_RATE={ LR } -e VAL_DATASET={ VAL_D } -e SAVE_EVERY={ SAVE_EVERY } \
-v {_local_dir_to_save_your_model_}:/home/storage/training \
-v {_local_dir_for_dataset}:/home/gpt-training/dataset \
--name dtp-training yqchenee1/dtp-training:{tag}
```
  * gpu
```bash
nvidia-docker run --privileged -itd \
-e RESTORE_FROM={ RF } -e LEARNING_RATE={ LR } -e VAL_DATASET={ VAL_D } -e SAVE_EVERY={ SAVE_EVERY } \
-v {_local_dir_to_save_your_model_}:/home/storage/training \
-v {_local_dir_for_dataset}:/home/gpt-training/dataset \
-v /usr/local/nvidia-driver/nvidia_driver/410.129/lib:/usr/local/nvidia/lib \
-v /usr/local/nvidia-driver/nvidia_driver/410.129/lib64:/usr/local/nvidia/lib64 \
--name dtp-training yqchenee1/dtp-training:{tag}
```
>  self-hosted?

----

# Run training
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python train.py --dataset dataset --save_every 1000 --model_name=ci_training --val_every=300 --run_name=training

# Improve and Questions
### how to count loss
### how long to train
- => see loss by time
  - -> converge or not
## OOM error
> use python.subprocess to call the program in the while loop
>> it seems there are no error in the program due to this case,
>> the program terminate normally and has a return code 0

---

## validation
-  need approve?:

---

## size
- val_batches: val_batch_count, val_batch_size, n_ctx
- label: val_batch_size, n_ctx
  - value = [0, n_vocab-1]
- model_output:
  - "logits": val_batch_size, n_ctx, n_vocab
  - use tf.nn.softmax(mo['logits']) to transfer into probability

---

## cost
- let n_ctx = 5
- let batch_size = 1
- ex:
```
model input = [5, 18, 55, 30, 101]
model output logits: 1* 5* 50257
transfer it into prob. => mo_prob
cost = -sigma[ ln(mo_prob[0][0][ model_input[1:] ] ) ] / (4*1)
```
```
model_input: 5 18 55 30 101
                ^  ^  ^  ^
                |  |  |  |
model_output:   a  b  c  d  e
```
> closer look at cost(val_cost) in train.py- validation()
```python
vo = model.model(hparams=hparams, X=np.stack(batch))
print("softmax")
print(vo['logits'][:, :-1])
print(vo['logits'][:, :-1].eval())
v_logits = np.array(vo['logits'][:, :-1].eval()) # 2, 1023, 50257
label = (np.array(batch)[:, 1:]) # 2, 1023
v_prop = tf.nn.softmax(vo['logits']) # 2, 1023, 50257

count_loss = [0, 0]
for i in range(2):
    for j in range(1023):
        print(j)
        count_loss[i] -= tf.log( v_prop[i][j][label[i][j]] )
```

---

## model.py
- batch = batch_size
- sequence = sequence_length (n_ctx)

---

## test
```python
import pytest
import requests
import re
import os
import random


def t_expected_output(url: str, text: str, expect: str):
    text = text.strip()
    response = requests.post(f"{url}/autocomplete", json={
        "text": text
    })
    if response.status_code > 200:
        assert False, "http request should success"
        return
    results: dict = response.json()
    assert "result" in results, "response should have key result"
    results: list[str] = results.get("result")
    found = False
    for result in results:
        if expect in result:
            found = True
            break
    return found


def t_files(url: str, filename: str):
    filename = "dataset/" + filename
    print(f"open {filename}")
    with open(filename, 'r', encoding='utf-8') as openFile:
        rule: str = openFile.read()
        words: list[str] = rule.split(' ')
    text = words[0] + ' '
    li_exp = 1
    num_correct = 0
    while True:
        print("text: ", text, "expect: ", words[li_exp])
        correct = t_expected_output(
            url,
            text,
            words[li_exp]
        )
        text = text + words[li_exp] + ' '
        if correct:
            num_correct += 1
        li_exp += 1
        if li_exp == len(words):
            break
    return (num_correct, len(words)-1)


def test_fit_rules(url: str):
    files: list = os.listdir('dataset/')
    print(f"there are {len(files)} files in dataset/")
    for i in range(5):
        file_no = random.randrange(len(files))
        nc, leng = t_files(url, files[file_no])
        print(nc, leng)
```
