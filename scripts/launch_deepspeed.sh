deepspeed \
  --num_gpus=8 \
  training/train.py \
  --deepspeed deepspeed/ds_zero2.json
