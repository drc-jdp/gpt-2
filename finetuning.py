import os
import sys
import subprocess

# PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python train.py
# --dataset drc_v3.npz --save_every 1000 --model_name=bpe-test
# --val_every=300 --run_name=run_v3 --learning_rate=0.00001
# --val_batch_count=60 --sample_every=200


def main():
    os.environ['PYTHONPATH'] = 'src'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logFile = open('log', 'w')
    logFile.write(
        "python train.py --dataset=drc_v3.npz \
        --save_every=1000 --model_name=bpe-test --val_every=300 \
        --val_batch_count=60 --sample_every=200 --run_name=run_v3 \
        --learning_rate=0.00001"
        )
    logFile.flush()
    t = 1
    while True:
        logFile.write(f'the {t}-th time')
        p = subprocess.call([
            "python", "train.py", "--dataset=drc_v3.npz",
            "--save_every=1000", "--model_name=bpe-test",
            "--val_every=300", "--val_batch_count=60",
            "--sample_every=200", "--run_name=run_v3",
            "--learning_rate=0.00001"
        ])
        logFile.write(f'error\n')
        logFile.flush()
        subprocess.call(["sleep", "5s"])


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print('kkkkkkkkkk', ex)
