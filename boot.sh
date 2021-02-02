cp models/ci_training/* /home/storage/training/
python train_mgr.py $1 --learning_rate=$2 --save_every=$3 \
    --val_dataset=$4 --val_every=$5 --val_batch_count=$6

