cp models/ci_training/* /home/storage/training/
echo $4
python train_mgr.py $1 --learning_rate=$2 --val_dataset=$4 --save_every=$3

