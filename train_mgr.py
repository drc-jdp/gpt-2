import os
import sys
import subprocess
import argparse

# PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python train.py
# --dataset dataset --save_every 1000 --model_name=ci_training
# --val_every=500 --val_batch_count=50 --sample_every=1000
# --restore_from=$1 --run_name=training --learning_rate=0.00002(default)

parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('restore_from', type=str, default='no', help='"latest", "fresh", "no"')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')


def main(args):
    os.environ['PYTHONPATH'] = 'src'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    with open('../storage/training/log', 'a') as logFile:
        logFile.write(
            f"python train.py --dataset=dataset\
            --save_every=1000 --model_name=ci_training\
            --val_every=500  --val_batch_count=50 --sample_every=1000\
            --restore_from={args.restore_from} --run_name=training --learning_rate={args.learning_rate}"
            )
    t = 1
    while True:
        p = subprocess.call([
            "python", "train.py", "--dataset=dataset",
            "--save_every=10", "--model_name=ci_training",
            "--val_every=500", "--val_batch_count=50",
            "--sample_every=1000", "--run_name=training",
            f"--learning_rate={args.learning_rate}",
            f"--restore_from={args.restore_from}"
        ])
        with open('../storage/training/log', 'a') as logFile:
            logFile.write(f'the {t}-th time error\n')
        subprocess.call(["sleep", "5s"])
        t += 1


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        main(args)
    except Exception as ex:
        print(ex)
