#!/bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=2

# Change for multinode config
MASTER_ADDR=10.144.0.109
NNODES=2
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Set default value for MASTER_PORT if not already set
MASTER_PORT=${MASTER_PORT:-120899}

DATA_PATH=/home/yn2161/ke/mlsys/fastMoE/examples/megatron/Megatron-LM-2.5/openwebtext_subset_text_document
CHECKPOINT_PATH=/home/yn2161/ke/mlsys/fastMoE/examples/megatron/Megatron-LM-2.5/experiment

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.run $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 3 \
    --hidden-size 8 \
    --num-attention-heads 4 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 1 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file vocab.json \
    --merge-file merges.txt \
    --data-impl mmap \
    --split 1,50,949 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --checkpoint-activations \
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --fp16 \
    --fmoefy \
    --num-experts 2\
    --top-k 1