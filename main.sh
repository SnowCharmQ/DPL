#!/bin/bash

METHOD="DPL"
NUM=(1 2 4 8)
CATEGORIES=("Movies_and_TV" "CDs_and_Vinyl" "Books")
DATASET="test"
OUTPUT_DIR="./output"
mkdir -p $OUTPUT_DIR

echo "$METHOD"
for i in "${NUM[@]}"; do
    for CATEGORY in "${CATEGORIES[@]}"; do
        echo "Category: $CATEGORY | Number of retrieved: $i"

        python model-infer.py \
            --method $METHOD \
            --gpu 0 \
            --dataset $DATASET \
            --category $CATEGORY \
            --output_dir $OUTPUT_DIR \
            --max_tokens 2048 \
            --num_retrieved $i \
            --retriever bm25 \
            --eval_batch_size 4 2>&1 | tee -a $OUTPUT_DIR/$METHOD-$CATEGORY.log

        python model-eval.py \
            --method $METHOD \
            --num $i \
            --category $CATEGORY \
            --output_dir $OUTPUT_DIR \
            --dataset $DATASET 2>&1 | tee -a $OUTPUT_DIR/$METHOD-$CATEGORY.log
            
    done
done