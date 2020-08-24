
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python train.py \
--dataset dataset --save_every 2 --model_name=ci_training \
--restore_from=$1 --sample_length=255 --run_name=training