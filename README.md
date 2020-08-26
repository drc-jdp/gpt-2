# Docker Usage
## model parameter
1.  --dataset { **dataset** } : the path to your data, both directory and file are ok
2.  --run_name { **rn** } : dir to save model => CHECKPOINT_DIR/rn
3.  --restore_from { **rf** } : 
    * no  : pre-train the model only with the param in models/hparams.json and vocab
    * latest  : train model with pre-trained model from CHECKPOINT_DIR/rf
    * fresh : train model with origin model from models/mn
4.  

## CI/CD
1. for **different model**, create new **branch** in GitHub
1. modify the hparams, vocab, encoder in {models/}
2. modify the model in train.py or src/model.py
3. push to GitHub, start CI/CD automatically
4. type on server 
```
docker run image -e RESTORE_FROM={ rf } -itd -v {_local_dir_to_save_your_model_}:/home/storage/training --name dtp-training yqchenee/dtp-training:{tag}
``` 
>  self-hosted?

----

# Run training
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python train.py --dataset dataset --save_every 1000 --model_name=ci_training --val_every=300 --run_name=training

# Improve and Question
how to count loss
how long to train
- => see loss by time
  - -> converge or not

---

## validation
-  need approve?:
  - use the same tokens all the time.

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

---

## model.py
- batch = batch_size
- sequence = sequence_length (n_ctx)

---
