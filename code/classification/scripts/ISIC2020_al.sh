for budget in high low
do
    for method in random uncertainty entropy margin coreset_L2 coreset_cosine dbal bald badge 
    do
        for seed in 1000 2000 3000 4000 5000
        do 
            CUDA_VISIBLE_DEVICES=0 python tools/train_al.py \
                --cfg configs/ISIC2020/al/RESNET18_${budget^^}_BUDGET.yaml \
                --al ${method} \
                --exp-name ${method}_${budget}_budget \
                --seed ${seed} 
        done
    done
done