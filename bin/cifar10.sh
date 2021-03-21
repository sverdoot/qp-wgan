#!/usr/bin/env bash

BATCH_SIZE=128
SEED=100
REG_COEF_1=0.1
REG_COEF_2=10

DATE=$(date +%H%M%S-%d%m)


for p in 1 2; do
    for q in 1 2; do
        for n_critic in 1 5; do
        
            echo q: ${q} p: ${p} critic iterations: ${n_critic} 
            
            PYTHONPATH=. python src/main.py \
                --task cifar10101010101010101010 \
                --q ${q} \
                --p ${p} \
                --n_critic_iter ${n_critic} \
                --batch_size ${BATCH_SIZE} \
                --seed ${SEED} \
                --reg_coef1 ${REG_COEF_1} \
                --reg_coef2 ${REG_COEF_2} \
                --search_space full 
        done
    done
done
