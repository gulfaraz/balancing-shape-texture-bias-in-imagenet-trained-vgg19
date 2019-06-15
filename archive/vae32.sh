for gamma in 0.0 10.0 50.0
do
    for zdim in 128 32
    do
        for beta in 0.0 0.2 1.0 10.0 100.0
        do
            python run.py --model nonstylized_vae$zdim\_beta$beta\_gamma$gamma --train --numberOfEpochs 50 --zdim $zdim --beta $beta --gamma $gamma --batchSize 128 --inputSize 128 &
        done
        wait
    done
done

