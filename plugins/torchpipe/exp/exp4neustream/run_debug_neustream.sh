
for slo_scale in "1.5" #"0.9" "0.95" "1"
do
    # for rate_scale in "0.75" "1.0" "1.25" "1.5" "1.75" "2.0" "2.25" "2.5"
    # for rate_scale in $(seq 1.0 0.5 7.0) 1.25
    # for rate_scale in "1" "2" "3" "4" "5"
    for rate_scale in "3"
    do
        echo "Running with variable rate: $rate_scale"
        USE_TRT="True" python SD_neustream.py --log "neustream_request500_ours_rate_log" --image_size 256 --rate_scale $rate_scale --cv_scale 0.5 --slo_scale $slo_scale --extra_vae_safety_time 0.06 --step_delta 0.95
    done
done