#!/bin/bash
# 在rtx4090/RTX4090 * 256/512下，有三种setting
if [ -z "$BASH" ]; then
    echo "Error: This script must be run with bash, not sh!" >&2
    exit 1
fi

# 此脚本跑rtx4090下的实验
# mkdir "neustream_request500_ours_cv_log"
# mkdir "neustream_request500_ours_slo_log"
# mkdir "neustream_request500_ours_rate_log"

# 对于image_size=512, 默认值使用更大的slo_factor

# extra_time_for_vae&safety
# 保留4090上的设定
# image_size=256时, 设定为0.06s
# image_size=512时, 设定为0.1s
for can_drop in '0' '1'
do
    for slo_scale in "7" #"0.9" "0.95" "1"
    do
        # for rate_scale in "0.75" "1.0" "1.25" "1.5" "1.75" "2.0" "2.25" "2.5"
        # for rate_scale in $(seq 1.0 0.5 7.0) 1.25
        # for slo_factor in "3" "3.5" "4" "4.5" "5" "6"   # 0.95 4.5   0.98 3.5
        # for slo_factor in "13" "14" "15"
        for rate_scale in "1" "2" "3" "4" "5"
        do
            echo "Running with variable rate: $rate_scale"
            USE_TRT="True" python SD_omniback.py --image_size 256 --rate_scale $rate_scale --cv_scale 0.5 --slo_scale $slo_scale --can_drop $can_drop 
        done
    done
done
