import os
import sys
import subprocess

# PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python train.py
# --dataset drc_v3.npz --save_every 1000 --model_name=pre-train
# --val_every=3000 --val_batch_count=20 --sample_every=1000
# --run_name=run_v3_no --learning_rate=0.00025


def main():
    os.environ['PYTHONPATH'] = 'src'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    logFile = open('log', 'w')
    logFile.write(
        "python train.py --dataset=drc_v3.npz \
        --save_every=1000 --model_name=pre-train --val_every=3000 \
        --val_batch_count=20 --sample_every=1000 --run_name=run_v3_no \
        --learning_rate=0.0003"
        )
    logFile.flush()
    t = 1
    while True:
        logFile.write(f'the {t}-th time')
        p = subprocess.call([
            "python", "train.py", "--dataset=drc_v3.npz",
            "--save_every=1000", "--model_name=pre-train",
            "--val_every=3000", "--val_batch_count=20",
            "--sample_every=1000", "--run_name=run_v3_no",
            "--learning_rate=0.0003"
        ])
        logFile.write(f'error\n')
        logFile.flush()
        subprocess.call(["sleep", "5s"])


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print('kkkkkkkkkk', ex)
