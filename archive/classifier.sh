for zdim in 48
do
    for gamma in 0.0
    do
        for beta in 0.0 0.2 1.0
        do
            python run.py --train --model nonstylized_classifier_z$zdim\_beta$beta\_gamma$gamma --inputSize 32 --learningRate 0.001 --zdim $zdim --beta $beta --gamma $gamma --vaeImageSize 32 --numberOfEpochs 5 &
        done
        wait
    done
done
