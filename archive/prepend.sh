for filename in *; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    if [[ $filename != *'.sh' ]]
    then
        echo mv $filename $1_$filename
    fi
done
