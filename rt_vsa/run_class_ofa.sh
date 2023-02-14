#!/bin/bash

export PYTHONPATH="`pwd`/../"
classes="810 782 470"
for iter in 1
do 
    for class in ${classes}
    do
        for mode in "per_class"
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
                --classes_list ${class} \
                --mode ${mode} | tee logs/${mode}_${iter}_${class}.log
        done
    done
done
