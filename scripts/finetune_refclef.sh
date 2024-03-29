mpirun -n 8 python tools/train_new.py \
--batch_size 8 \
--trainable_layers 3 \
--max_epochs 20 \
--drop_epochs 10 \
--norm_layer frozen_bn \
--dataset_type refclef \
--splitBy berkeley \
--translate \
--multi_scale \
--lr_base 1e-4 \
--lr_visual 1e-5 \
--lr_lang 1e-5 \
--arch_loss_coef 0.0 \
--pretrained_path weights/pretrain-weights/new_lads_pretrained.ckpt \
--comments "finetune_refclef"