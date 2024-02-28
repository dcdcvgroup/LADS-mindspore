mpirun -n 8 python tools/train_new.py \
--batch_size 8 \
--norm_layer frozen_bn \
--dataset_type refcoco+ \
--splitBy unc \
--translate \
--multi_scale \
--pretrained_path {PRETRAINED_FILE} \
--comments "eval_refcoco+"