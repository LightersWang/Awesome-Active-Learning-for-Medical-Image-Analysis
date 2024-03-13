for budget in 10
do
    for active_method in random softmax_confidence softmax_margin softmax_entropy core_set_L2 core_set_cosine badge
    do 
        for seed in 1000 2000 3000 4000 5000
        do
            python train_al.py \
                --exp_dir exps/ACDC/al \
                --seed ${seed} \
                --num_workers 4 \
                --dataset ACDC \
                --active_method ${active_method} \
                --count_per_round ${budget} \
                --model unet_plain \
                --unet_channels 32 64 128 256 512 \
                --dropout_prob 0.1 \
                --normalization batch \
                --num_classes 4 \
                --img_size 256 \
                --optimizer adam \
                --lr_schedule poly \
                --lr 0.001 \
                --weight_decay 5e-4 \
                --train_batch_size 32 \
                --total_itrs 4000 \
                --val_mode volume \
                --val_period 1000 \
                --val_batch_size 1
        done
    done
done
