#!/usr/bin/env bash

BATCH_SIZE=300
SEED=100
REG_COEF_1=0
REG_COEF_2=0

DATE=$(date +%H%M%S-%d%m)
mkdir -p ../models


for p in 1 2; do
    for q in 1 2; do
        for n_critic in 1 5; do
        
            echo q: ${q} p: ${p} critic iterations: ${n_critic} 
            
            PYTHONPATH=. python src/main.py \
                --task mnist \
                --q ${q} \
                --p ${p} \
                --n_critic_iter ${n_critic} \
                --batch_size ${BATCH_SIZE} \
                --seed ${SEED} \
                --reg_coef1 ${REG_COEF_1} \
                --reg_coef2 ${REG_COEF_2} \
                --search_space x 

        done
    done
done


