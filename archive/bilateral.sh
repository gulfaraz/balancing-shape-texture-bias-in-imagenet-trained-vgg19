for filename in *; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    if [[ $filename != *'.sh' && $filename == *'_bilateral.ckpt' ]]
    then
        echo mv $filename bilateral_${filename:0:-15}.ckpt
    fi
done
