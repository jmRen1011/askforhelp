deepspeed --include localhost:3 --master_port=29503 fuse_main.py \
  --output_dir mistral-add=True_cross=True-scenario=False-0528-0129-5epoch1111 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 8 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --weight_decay 1e-3 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 5 \
  --max_grad_norm 1.0 \
  --evaluation_strategy steps \
  --eval_steps 10 \
  --per_device_eval_batch_size 1 \
  --load_best_model_at_end True \
  --metric_for_best_model f1_micro \
  --greater_is_better True \
  --load_checkpoint False \
  --predict_with_generate True \
  --deepspeed configs/deepspeed_config_stage3.json \
  # --checkpoint_path /raid_sda/home/rjm/mine/mistral-add=True_cross=True-scenario=False-0528-0129-5epoch/checkpoint-10310 \

# deepspeed --include localhost:0,1 fuse_main.py \
#   --output_dir mistral-cross-with-scenario-0513-1941-4epoch \
#   --num_train_epochs 4 \
#   --per_device_train_batch_size 1 \
#   --learning_rate 1e-4 \
#   --gradient_checkpointing False \
#   --gradient_accumulation_steps 8 \
#   --bf16 True \
#   --tf32 True \
#   --lr_scheduler_type "constant_with_warmup" \
#   --logging_steps 1 \
#   --save_steps 100 \
#   --save_total_limit 20 \
#   --deepspeed configs/deepspeed_config_stage3.json
  # --load_checkpoint True 
  # --checkpoint_path "./mistral-fuse-new-load-4epoch-qv-lora-save/checkpoint-300"

# torchrun --nproc_per_node 4 main.py \
#   --output_dir mixtral-moe-lora-instruct-sharc \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 1 \
#   --learning_rate 1e-4 \
#   --gradient_checkpointing True \
#   --gradient_accumulation_steps 8 \
#   --bf16 True \
#   --tf32 True \
#   --lr_scheduler_type "constant_with_warmup" \
#   --logging_steps 25 \
#   --save_steps 100 \
#   --save_total_limit 3 \
#   --deepspeed configs/deepspeed_config_stage3.json

#   --model_id tiiuae/falcon-180B \
#   --dataset_path dolly-processed \