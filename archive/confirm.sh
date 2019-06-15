for model in vgg19_in_all_tune_all_bilateral_confirm \
    vgg19_bn_in_single_tune_all_confirm_again \
    vgg19_in_all_tune_all_confirm_again \
    vgg19_in_single_tune_all_confirm_again \
    vgg19_in_affine_single_tune_all_confirm_again \
    vgg19_in_sm_all_tune_all_confirm_again \
    vgg19_bn_in_single_tune_all_bilateral_confirm \
    vgg19_in_single_tune_all_bilateral_confirm \
    vgg19_in_affine_single_tune_all_bilateral_confirm \
    vgg19_in_sm_all_tune_all_bilateral_confirm \
    vgg19_vanilla_tune_fc_bilateral_confirm_again \
    vgg19_bn_all_tune_fc_bilateral_confirm_again \
    vgg19_bn_in_single_tune_all_bilateral_confirm_again \
    vgg19_in_all_tune_all_bilateral_confirm_again \
    vgg19_in_single_tune_all_bilateral_confirm_again \
    vgg19_in_affine_single_tune_all_bilateral_confirm_again \
    vgg19_in_sm_all_tune_all_bilateral_confirm_again
do
    srun --gres=gpu:1 --constraint=TitanX --time=48:00:00 python confirm.py --train --model $model
done
