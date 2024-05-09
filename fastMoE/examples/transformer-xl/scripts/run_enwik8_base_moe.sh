#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 trainPipelinetest.py \
        --cuda \
        --data ../one-billion-words/ \
        --dataset lm1b \
        --n_layer 3 \
        --d_model 8 \
        --n_head 2 \
        --d_head 4 \
        --d_inner 32 \
        --dropout 0.5 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.25 \
        --warmup_step 0 \
        --max_step 400000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 16 \
        --gpu0_bsz 4 \
        --multi_gpu \ 
        --moe --moe-num-expert 8 --moe-top-k 2 
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 eval.py \
        --cuda \
        --data ../one-billion-words/ \
        --dataset lm1b \
        --tgt_len 512 \
        --mem_len 512 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
