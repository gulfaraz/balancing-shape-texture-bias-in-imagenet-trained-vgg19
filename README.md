# BALANCING SHAPE-TEXTURE BIAS IN IMAGENET-TRAINED VGG19

This repository holds code used to run experiments for my [2019 Masters A.I. thesis research](http://gulfaraz.com/share/balancing_shape_texture_bias_master_thesis_gulfaraz_rahman.pdf) at the [University of Amsterdam](https://www.uva.nl/).

[Thesis report is publicly available for download.](http://gulfaraz.com/share/balancing_shape_texture_bias_master_thesis_gulfaraz_rahman.pdf)

[Presentation is publicly available on Google Drive.](https://docs.google.com/presentation/d/1acGki6BS219MIwn5HYOvX3zifaY5q9uybvtGzztxxAE/edit?usp=sharing)

---
## Requirements
[Anaconda](https://www.anaconda.com/distribution/#download-section)

Then create a conda environment with [torchenv.yml](./torchenv.yml) using,

`conda env create -f torchenv.yml`

`torchenv` conda environment should have the required packages to run the experiments.

---
## Dataset

**ImageNet200** - a subset of [ImageNet](http://image-net.org/download) was used to train and validate the models.

Use the `.txt` files in [ImageNet200 folder](./ImageNet200) to create the dataset.

Use [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet) to create the necessary stylized datasets.

---
## Model Training

To train **Vanilla** model on non-stylized ImageNet200,

`python run.py --model nonstylized_vgg19_vanilla_tune_fc --train`

To train **Single IN** model on non-stylized ImageNet200,

`python run.py --model nonstylized_vgg19_in_single_tune_all --train`

To train **Vanilla** model on non-stylized ImageNet200 with bilateral filter,

`python run.py --model nonstylized_bilateral_vgg19_vanilla_tune_fc --bilateral --train`

To train **VAE** model with stylized target,

`python run.py --model stylized_vae1024_beta0.2_gamma50.0 --numberOfEpochs 100 --zdim 1024 --beta 0.2 --gamma 50.0 --batchSize 32 --inputSize 128 --vaeImageSize 128 --dataset stylized --train`

To train **Single IN** model with stylized latent representation as auxiliary signal,

`python run.py --model stylized_latent_vgg19_in_single_tune_all --zdim 1024 --beta 0.2 --gamma 50.0 --batchSize 32 --dataset stylized --train`

---
## Model Evaluation

Run the same commands as training without the `train` flag.

To test **Vanilla** model,

`python run.py --model nonstylized_vgg19_vanilla_tune_fc`

---
## Command Line Arguments

```
usage: run.py [-h] [--rootPath ROOTPATH] [--numberOfWorkers NUMBEROFWORKERS]
              [--bilateral] [--dataset {nonstylized,stylized,highpass}]
              [--disableCuda] [--cudaDevice CUDADEVICE]
              [--torchSeed TORCHSEED] [--inputSize INPUTSIZE]
              [--vaeImageSize VAEIMAGESIZE] [--numberOfEpochs NUMBEROFEPOCHS]
              [--batchSize BATCHSIZE] [--learningRate LEARNINGRATE]
              [--autoencoderLearningRate AUTOENCODERLEARNINGRATE]
              [--classifierLearningRate CLASSIFIERLEARNINGRATE] [--beta BETA]
              [--zdim ZDIM] [--gamma GAMMA] [--train] [--exists]
              [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --rootPath ROOTPATH   output path (default: /var/node433/local/gulfaraz)
  --numberOfWorkers NUMBEROFWORKERS
                        number of threads used by data loader (default: 8)
  --bilateral           apply bilateral filter at input layer (default: False)
  --dataset {nonstylized,stylized,highpass}
                        name of dataset to use for training (default:
                        nonstylized)
  --disableCuda         disable the use of CUDA (default: False)
  --cudaDevice CUDADEVICE
                        specify which GPU to use (default: 0)
  --torchSeed TORCHSEED
                        set a torch seed (default: 42)
  --inputSize INPUTSIZE
                        extent of input layer in the network (default: 224)
  --vaeImageSize VAEIMAGESIZE
                        extent of input and target layer in the autoencoder
                        (default: 128)
  --numberOfEpochs NUMBEROFEPOCHS
                        number of epochs for training (default: 50)
  --batchSize BATCHSIZE
                        batch size for training (default: 32)
  --learningRate LEARNINGRATE
                        learning rate for training (default: 0.0001)
  --autoencoderLearningRate AUTOENCODERLEARNINGRATE
                        learning rate for autoencoder training (default:
                        0.001)
  --classifierLearningRate CLASSIFIERLEARNINGRATE
                        learning rate for classifier training (default: 0.001)
  --beta BETA           beta value for the betavae loss (default: 0.2)
  --zdim ZDIM           latent space dimension size for the betavae (default:
                        128)
  --gamma GAMMA         weight of the classification loss in vae (default:
                        0.0)
  --train               train the models (default: False)
  --exists              check if the trained models exist (default: False)
  --model MODEL         name of model(s) (default: None)
  ```

  ---

  Please feel free to create a GitHub issue for queries and bugs.