
for step_delta in "0.95" #"0.9" "0.95" "1"
do
    # for rate_scale in "0.75" "1.0" "1.25" "1.5" "1.75" "2.0" "2.25" "2.5"
    # for rate_scale in $(seq 1.0 0.5 7.0) 1.25
    for rate_scale in "4"   # 0.95 4.5   0.98 3.5
    do
        echo "Running with variable rate: $rate_scale"
        USE_TRT="True" python SD_neustream.py --log "neustream_request500_ours_rate_log" --image_size 256 --rate_scale $rate_scale --cv_scale 0.5 --slo_scale 7 --extra_vae_safety_time 0.06 --profile_device "rtx4090" --step_delta $step_delta
    done
done