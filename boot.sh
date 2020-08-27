
# PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python train.py \
# --dataset dataset --save_every 200 --model_name=ci_training \
# --restore_from=$1 --run_name=training

if [ -z "$2" ]; then
    python train_mgr.py $1
else
    python train_mgr.py $1 --learning_rate=$2
fi
