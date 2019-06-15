for model in stylized_vgg19_vanilla_tune_fc_confirm \
stylized_vgg19_vanilla_tune_fc_confirm_again
do
    echo srun --gres=gpu:1 --constraint=TitanX --time=48:00:00 python run.py --train --model $model --dataset stylized
done
wait
