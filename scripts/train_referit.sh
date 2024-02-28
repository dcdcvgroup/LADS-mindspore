mpirun -n 8 python tools/train_new.py \
--batch_size 8 \
--batch_sum 1024 \
--trainable_layers 3 \
--selector_bias 3.0 \
--norm_layer frozen_bn \
--max_epochs 120 \
--drop_epochs 90 \
--drop_rate 0.1 \
--eval_step 1 \
--warmup_steps 4000 \
--dataset_type refclef \
--splitBy berkeley \
--translate \
--multi_scale \
--arch_loss_coef 0.1 \
--lr_base 4e-4 \
--lr_visual 4e-5 \
--lr_lang 4e-5 \
--comments "train_arch_coef_0.1_bias_init_3_batch_1024" \
--pretrained_path ""
# 目标69.76
# resnet权重应该没有问题
# 尝试一下增大学习率*2,和使用warmup=150，提到了68.22
# 尝试增大trainable_layers为4，反而降低了 fail
# warmup_steps 32000, fail
## 尝试增大selector_bias fail
## 把arch_loss_coef降低（在finetune中这样是有效的）fail
# 尝试早点drop, 调整drop_rate,fail
# 降低trainable_layers, fail
# 增大batch_sum, 有效
# 增大arch_loss_coef到0.15, fail
# warmup_steps不能太大,尽量减小