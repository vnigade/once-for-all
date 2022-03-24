#!/bin/bash

export PYTHONPATH="`pwd`/../"

for iter in 1 2
do 
    for mode in "per_class" "all_classes"
    do
        python3 test_class_ofa.py \
            --calib_bn_dataset "./imagenet_data/ofa" \
            --dataset_path_all_classes "./imagenet_data/ofa" \
            --dataset_path_per_class "./imagenet_data/tench" \
            --random_search_iter 10 \
            --mode ${mode} | tee logs/${mode}_${iter}.log
    done
done