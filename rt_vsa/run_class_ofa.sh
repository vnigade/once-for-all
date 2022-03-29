#!/bin/bash

export PYTHONPATH="`pwd`/../"

for iter in 1
do 
    for mode in "all_classes"
    do
        if [ ${mode} = "per_class" ]; then
            search_dataset_path="./imagenet_data/full"
        else
            # Use only the subset of the full dataset.
            search_dataset_path="./imagenet_data/ofa"
        fi

        python3 test_class_ofa.py \
            --calib_bn_dataset "./imagenet_data/ofa" \
            --search_dataset_path ${search_dataset_path} \
            --test_dataset_path "./imagenet_data/full" \
            --random_search_iter 1000 \
            --mode ${mode} | tee logs/${mode}_${iter}.log
    done
done
