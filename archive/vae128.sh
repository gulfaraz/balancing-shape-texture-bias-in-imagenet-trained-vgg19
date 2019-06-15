for gamma in 1.0 10.0 100.0 200.0
do
    for zdim in 48
    do
        for beta in 0.2
        do
            echo python run.py --model nonstylized_vaegamma$zdim\_beta$beta\_gamma$gamma --train --numberOfEpochs 10 --zdim $zdim --beta $beta --gamma $gamma --batchSize 32 --inputSize 32 --vaeImageSize 32 &
        done
    done
done
wait
