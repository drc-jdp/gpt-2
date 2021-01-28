cp models/ci_training/* /home/storage/training/
python train_mgr.py $1 --learning_rate=$2 --val_dataset=$3 --save_every=$4

