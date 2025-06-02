#!/bin/bash

METHOD="DPL"
NUMS=(8)
CATEGORIES=("Movies_and_TV" "CDs_and_Vinyl" "Books")
DATASET="test"
OUTPUT_DIR="./output"

mkdir -p $OUTPUT_DIR

echo "$METHOD"
for NUM in "${NUMS[@]}"; do
    for CATEGORY in "${CATEGORIES[@]}"; do
        echo "Category: $CATEGORY | Number of retrieved: $NUM"

        python model-infer.py \
            --method $METHOD \
            --gpu 0 \
            --dataset $DATASET \
            --category $CATEGORY \
            --output_dir $OUTPUT_DIR \
            --max_tokens 2048 \
            --num_retrieved $NUM \
            --retriever bm25

        python eval-basic.py \
            --method $METHOD \
            --num $NUM \
            --category $CATEGORY \
            --output_dir $OUTPUT_DIR \
            --dataset $DATASET

        python eval-72B.py \
            --method $METHOD \
            --num $NUM \
            --category $CATEGORY \
            --output_dir $OUTPUT_DIR \
            --dataset $DATASET
            
    done
done