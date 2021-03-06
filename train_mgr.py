import os
import sys
import subprocess
import argparse


parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('restore_from', type=str, default='no', help='"latest", "fresh", "no"')
parser.add_argument('--learning_rate', type=float, default=0.00002, help='Learning rate for Adam')
parser.add_argument('--save_every', type=int, default=1000)
parser.add_argument('--val_dataset', type=str, default=None)
parser.add_argument('--val_every', type=int) # default: 100  
parser.add_argument('--val_batch_count', type=int) # default: 20


def main(args):
    save_every = args.save_every
    os.environ['PYTHONPATH'] = 'src'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    t = 1
    p = subprocess.call([
            "python", "train.py", "--dataset=dataset",
            f"--save_every={save_every}", "--model_name=ci_training",
            f"--val_every={args.val_every}", f"--val_batch_count={args.val_batch_count}",
            f"--val_dataset={args.val_dataset}",
            "--sample_every=1000", "--run_name=training",
            f"--learning_rate={args.learning_rate}",
            f"--restore_from={args.restore_from}"
    ])
    while True:
        p = subprocess.call([
            "python", "train.py", "--dataset=dataset",
            f"--save_every={save_every}", "--model_name=ci_training",
            f"--val_every={args.val_every}", f"--val_batch_count={args.val_batch_count}",
            f"--val_dataset={args.val_dataset}",
            "--sample_every=1000", "--run_name=training",
            f"--learning_rate={args.learning_rate}"
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
