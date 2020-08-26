
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python train.py \
--dataset dataset --save_every 200 --model_name=ci_training \
--restore_from=$1 --run_name=training
