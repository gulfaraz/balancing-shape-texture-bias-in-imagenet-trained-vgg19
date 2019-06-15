# In[1]: Load Libraries

import os
import torch
import numpy as np
from score import *
from utils import *
from torchvision.utils import save_image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, 200))
np.random.shuffle(colors)

DEBUG = False

# In[2]: Supporting Functions

def calculate_similarity_loss(similarity, kernel_sizes=[64., 128., 256., 512., 512.]):
    number_of_kernels = torch.tensor(kernel_sizes)
    style_weights = torch.tensor([1e3/n**2 for n in number_of_kernels])
    weighted_similarity = style_weights * similarity
    return weighted_similarity.mean() + (number_of_kernels * style_weights).sum()


def calculate_reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0
    sigmoid = torch.nn.Sigmoid()
    mse_loss = torch.nn.MSELoss(reduction='sum')
    bce_logit_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

    if distribution == 'bernoulli':
        recon_loss = bce_logit_loss(x_recon, x).div(batch_size)
    elif distribution == 'gaussian':
        recon_loss = mse_loss(x_recon, x).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def calculate_kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# def calculate_reconstruction_loss(x, x_recon, distribution):
#     batch_size = x.size(0)
#     assert batch_size != 0
#     sigmoid = torch.nn.Sigmoid()
#     mse_loss = torch.nn.MSELoss(reduction='mean')
#     bce_logit_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

#     if distribution == 'bernoulli':
#         recon_loss = bce_logit_loss(x_recon, x)#.div(batch_size)
#     elif distribution == 'gaussian':
#         recon_loss = mse_loss(x_recon, x)#.div(batch_size)
#     else:
#         recon_loss = None

#     return recon_loss


# def calculate_kl_divergence(mu, logvar):
#     batch_size = mu.size(0)
#     assert batch_size != 0
#     if mu.data.ndimension() == 4:
#         mu = mu.view(mu.size(0), mu.size(1))
#     if logvar.data.ndimension() == 4:
#         logvar = logvar.view(logvar.size(0), logvar.size(1))

#     klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
#     total_kld = klds.sum(1).mean(0, True)
#     dimension_wise_kld = klds.mean(0)
#     mean_kld = klds.mean(1).mean(0, True)

#     return mean_kld, dimension_wise_kld, total_kld

# In[3]: Classifier

def validate(model, dataloader, criterion, logger, device, similarity_weight=None):
    logger.debug('Validation Start')
    model.eval()
    
    total_top1, total_top5, total_, top1_score, top5_score = 0, 0, 0, 0, 0
    loss = []
    if similarity_weight is not None:
        classification_loss = []
        similarity_loss = []

    for batch_index, batch in enumerate(dataloader):
        if similarity_weight is not None:
            output, batch_similarity = model(batch[dataloader.dataset.INDEX_IMAGE].to(device))
        else:
            output = model(batch[dataloader.dataset.INDEX_IMAGE].to(device))
        target = batch[dataloader.dataset.INDEX_TARGET].to(device)

        _, predicted_class = output.topk(5, 1, True, True)
        top1, top5, total = score(predicted_class, target)

        total_top1 += top1
        total_top5 += top5
        total_ += total

        # loss
        if similarity_weight is not None:
            batch_classification_loss = criterion(output, target)
            batch_similarity_loss = calculate_similarity_loss(batch_similarity)
        
            batch_loss = batch_classification_loss + (similarity_weight * batch_similarity_loss)
        else:
            batch_loss = criterion(output, target)

        loss.append(batch_loss.item())
        mean_loss = np.mean(loss)

        if similarity_weight is not None:
            classification_loss.append(batch_classification_loss.item())
            similarity_loss.append(batch_similarity_loss.item())
            mean_classification_loss = np.mean(classification_loss)
            mean_similarity_loss = np.mean(similarity_loss)

        top1_score = score_value(total_top1, total_)
        top5_score = score_value(total_top5, total_)
        if (batch_index + 1) % 10 == 0:
            if similarity_weight is not None:
                logger.debug('Validation Batch {}/{}: Top1 Accuracy {:.4f} Top5 Accuracy {:.4f}'.format(
                    batch_index + 1, len(dataloader), top1_score, top5_score) \
                    + ' Loss {:.4f} Classification Loss {:.4f} Similarity Loss {:.4f} Similarity Weight {:.2f}'.format(
                        mean_loss, mean_classification_loss, mean_similarity_loss, similarity_weight))
            else:
                logger.debug('Validation Batch {}/{}: Top1 Accuracy {:.4f} Top5 Accuracy {:.4f} Loss {:.4f}'.format(
                    batch_index + 1, len(dataloader), top1_score, top5_score, mean_loss))
            if DEBUG:
                break

    logger.debug('Validation End')
    return top1_score, top5_score, mean_loss


def train(model, dataloader, criterion, optimizer, logger, device, similarity_weight=None, grad_clip_norm_value=50):
    logger.debug('Training Start')
    model.train()

    total_top1, total_top5, total_, top1_score, top5_score = 0, 0, 0, 0, 0
    loss = []
    if similarity_weight is not None:
        classification_loss = []
        similarity_loss = []

    for batch_index, batch in enumerate(dataloader):
        optimizer.zero_grad()
        if similarity_weight is not None:
            output, batch_similarity = model(batch[dataloader.dataset.INDEX_IMAGE].to(device))
        else:
            output = model(batch[dataloader.dataset.INDEX_IMAGE].to(device))
        target = batch[dataloader.dataset.INDEX_TARGET].to(device)

        # accuracy
        _, predicted_class = output.topk(5, 1, True, True)
        top1, top5, total = score(predicted_class, target)

        total_top1 += top1
        total_top5 += top5
        total_ += total

        # loss
        if similarity_weight is not None:
            batch_similarity_loss = calculate_similarity_loss(batch_similarity)
            batch_classification_loss = criterion(output, target)

            batch_loss = batch_classification_loss + (similarity_weight * batch_similarity_loss)
        else:
            batch_loss = criterion(output, target)

        loss.append(batch_loss.item())

        # backprop
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_value)
        optimizer.step()
        
        # use mean metrics
        mean_loss = np.mean(loss)

        if similarity_weight is not None:
            classification_loss.append(batch_classification_loss.item())
            similarity_loss.append(batch_similarity_loss.item())
            mean_classification_loss = np.mean(classification_loss)
            mean_similarity_loss = np.mean(similarity_loss)

        top1_score = score_value(total_top1, total_)
        top5_score = score_value(total_top5, total_)
            
        if (batch_index + 1) % 10 == 0:
            if similarity_weight is not None:
                logger.debug('Training Batch {}/{}: Top1 Accuracy {:.4f} Top5 Accuracy {:.4f}'.format(
                    batch_index + 1, len(dataloader), top1_score, top5_score) \
                    + 'Loss {:.4f} Classification Loss {:.4f} Similarity Loss {:.4f} Similarity Weight {:.2f}'.format(
                        mean_loss, mean_classification_loss, mean_similarity_loss, similarity_weight))
            else:
                logger.debug('Training Batch {}/{}: Top1 Accuracy {:.4f} Top5 Accuracy {:.4f} Loss {:.4f}'.format(
                    batch_index + 1, len(dataloader), top1_score, top5_score, mean_loss))
            if DEBUG:
                break

    logger.debug('Training End')
    return top1_score, top5_score, mean_loss


def run(model_name, model, model_directory, number_of_epochs, learning_rate, logger,
        train_loader, val_loader, device, similarity_weight=None,
        dataset_names=['stylized-imagenet200-0.0', 'stylized-imagenet200-1.0'], load_data=None):
    checkpoint_path = pathJoin(model_directory, '{}.ckpt'.format(model_name))
    print(checkpoint_path)

    parameters = model.parameters()
    if 'classifier' in model_name:
        parameters = model.classifier.parameters()

    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, min_lr=1e-5)
    
    last_epoch = 0
    best_validation_accuracy = -1.0

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        last_epoch = checkpoint['epoch']
        best_validation_accuracy = checkpoint['validation_top5_accuracy']
        model.load_state_dict(checkpoint['weights'])
        optimizer.load_state_dict(checkpoint['optimizer_weights'])

    last_epoch += 1
    logger.info('Training model {} from epoch {}'.format(checkpoint_path, last_epoch))

    logger.info('Epochs {}'.format(number_of_epochs))
    logger.info('Batch Size {}'.format(train_loader.batch_size))
    logger.info('Number of Workers {}'.format(train_loader.num_workers))
    logger.info('Optimizer {}'.format(optimizer))
    logger.info('Learning Rate {}'.format(learning_rate))
    logger.info('Similarity Weight {}'.format(similarity_weight))
    logger.info('Device {}'.format(device))

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(last_epoch, number_of_epochs + 1):
        train_top1_accuracy, train_top5_accuracy, train_loss = train(
            model, train_loader, criterion, optimizer,
            logger, device, similarity_weight)
        validation_top1_accuracy, validation_top5_accuracy, validation_loss = validate(
            model, val_loader, criterion,
            logger, device, similarity_weight)
        logger.info('Epoch {}: Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
            epoch, train_loss, train_top1_accuracy, train_top5_accuracy) \
            + ' Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
                validation_loss, validation_top1_accuracy, validation_top5_accuracy))

        lr_scheduler.step(validation_loss)

        if validation_top5_accuracy > best_validation_accuracy:
            logger.debug('Improved Validation Score, saving new weights')
            os.makedirs(model_directory, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'train_top1_accuracy': train_top1_accuracy,
                'train_top5_accuracy': train_top5_accuracy,
                'train_loss': train_loss,
                'validation_top1_accuracy': validation_top1_accuracy,
                'validation_top5_accuracy': validation_top5_accuracy,
                'validation_loss': validation_loss,
                'weights': model.state_dict(),
                'optimizer_weights': optimizer.state_dict()
            }
            torch.save(checkpoint, pathJoin(model_directory, '{}.ckpt'.format(model_name)))
            best_validation_accuracy = validation_top5_accuracy

    logger.info('Epoch {}'.format(checkpoint['epoch']))

    evaluate_model(model_name, model, load_data, dataset_names,
        logger.info, similarity_weight is not None, device)
    logger.info('Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
        checkpoint['train_loss'], checkpoint['train_top1_accuracy'], checkpoint['train_top5_accuracy']))
    logger.info('Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
        checkpoint['validation_loss'], checkpoint['validation_top1_accuracy'], checkpoint['validation_top5_accuracy']))


# In[4]: Autoencoder

def plot_manifold(all_mu, all_class, manifold_filename):
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(all_mu)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    plt.clf()
    plt.scatter(x, y, color=[ colors[i] for i in all_class ])
    plt.savefig(manifold_filename, bbox_inches='tight')

def validate_autoencoder(model, loader, logger, device, reconstruction_grid_filename, manifold_filename, beta, gamma, criterion, save_reconstruction, distribution):
    logger.debug('Validation Start')
    model.eval()

    total_top1, total_top5, total_, top1_score, top5_score = 0, 0, 0, 0, 0
    loss = []
    all_mu = []
    all_class = []

    for batch_index, batch in enumerate(loader):
        batch_input = batch[loader.dataset.INDEX_IMAGE].to(device)
        batch_classification_target = batch[loader.dataset.INDEX_TARGET].to(device)
        batch_reconstruction_target = batch[loader.dataset.INDEX_TARGET_IMAGE].to(device)
        batch_class_prediction, batch_reconstruction, mu, logvar = model(batch_input)

        # accuracy
        _, predicted_class = batch_class_prediction.topk(5, 1, True, True)
        top1, top5, total = score(predicted_class, batch_classification_target)

        total_top1 += top1
        total_top5 += top5
        total_ += total

        # loss
        classification_loss = criterion(batch_class_prediction, batch_classification_target)
        reconstruction_loss = calculate_reconstruction_loss(batch_reconstruction_target, batch_reconstruction, distribution)
        total_kld, dim_wise_kld, mean_kld = calculate_kl_divergence(mu, logvar)
        effective_kl = beta * total_kld
        effective_classification_loss = gamma * classification_loss
        batch_loss = reconstruction_loss + effective_kl + effective_classification_loss

        logger.debug('Batch Loss: {}'.format(batch_loss.item()) \
            + ' Reconstruction Loss: {:.4f}'.format(reconstruction_loss.item()) \
            + ' Effective-KL {:.4f}'.format(effective_kl.item()) \
            + ' Effective-CE {:.4f}'.format(effective_classification_loss.item()) \
            + ' Cross-Entropy {:.4f}'.format(classification_loss.item()) \
            + ' KL-Divergence: {:.4f}'.format(total_kld.item()) \
            + ' Mean-KLD: {:.4f}'.format(mean_kld.item()))

        loss.append(batch_loss.item())

        mean_loss = np.mean(loss)

        top1_score = score_value(total_top1, total_)
        top5_score = score_value(total_top5, total_)

        if save_reconstruction:
            if batch_index == 0:
                n = min(batch_reconstruction_target.size(0), 8)
                comparison = torch.cat([batch_reconstruction_target[:n],
                    batch_reconstruction.view(*batch_reconstruction_target.shape)[:n]])
                save_image(comparison.cpu(), reconstruction_grid_filename, nrow=n, normalize=False)
            
            all_mu.append(mu)
            all_class.append(batch_classification_target)

        if (batch_index + 1) % 10 == 0:
            logger.debug('Validation Batch {}/{}: Loss {:.4f}'.format(batch_index + 1, len(loader), mean_loss) \
                + ' Top1 Accuracy {:.4f} Top5 Accuracy {:.4f}'.format(top1_score, top5_score))
            if DEBUG:
                break

    if save_reconstruction:
        all_mu = torch.cat(all_mu, dim=0).detach().cpu().numpy()
        all_class = torch.cat(all_class, dim=0).detach().cpu().numpy()
        plot_manifold(all_mu, all_class, manifold_filename)
    logger.debug('Validation End')
    return top1_score, top5_score, mean_loss

def train_autoencoder(model, loader, optimizer, logger, device, beta, gamma, criterion, distribution, grad_clip_norm_value=50):
    logger.debug('Training Start')
    model.train()

    total_top1, total_top5, total_, top1_score, top5_score = 0, 0, 0, 0, 0
    loss = []

    for batch_index, batch in enumerate(loader):
        optimizer.zero_grad()
        batch_input = batch[loader.dataset.INDEX_IMAGE].to(device)
        batch_classification_target = batch[loader.dataset.INDEX_TARGET].to(device)
        batch_reconstruction_target = batch[loader.dataset.INDEX_TARGET_IMAGE].to(device)
        batch_class_prediction, batch_reconstruction, mu, logvar = model(batch_input)

        # accuracy
        _, predicted_class = batch_class_prediction.topk(5, 1, True, True)
        top1, top5, total = score(predicted_class, batch_classification_target)

        total_top1 += top1
        total_top5 += top5
        total_ += total

        # loss
        classification_loss = criterion(batch_class_prediction, batch_classification_target)
        reconstruction_loss = calculate_reconstruction_loss(batch_reconstruction_target, batch_reconstruction, distribution)
        total_kld, dim_wise_kld, mean_kld = calculate_kl_divergence(mu, logvar)
        effective_kl = beta * total_kld
        effective_classification_loss = gamma * classification_loss
        batch_loss = reconstruction_loss + effective_kl + effective_classification_loss

        logger.debug('Batch Loss: {}'.format(batch_loss.item()) \
            + ' Reconstruction Loss: {:.4f}'.format(reconstruction_loss.item()) \
            + ' Effective-KL {:.4f}'.format(effective_kl.item()) \
            + ' Effective-CE {:.4f}'.format(effective_classification_loss.item()) \
            + ' Cross-Entropy {:.4f}'.format(classification_loss.item()) \
            + ' KL-Divergence: {:.4f}'.format(total_kld.item()) \
            + ' Mean-KLD: {:.4f}'.format(mean_kld.item()))

        loss.append(batch_loss.item())

        # backprop
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_value)
        optimizer.step()
        
        # use mean metrics
        mean_loss = np.mean(loss)

        top1_score = score_value(total_top1, total_)
        top5_score = score_value(total_top5, total_)
            
        if (batch_index + 1) % 10 == 0:
            logger.debug('Training Batch {}/{}: Loss {:.4f}'.format(batch_index + 1, len(loader), mean_loss) \
                + ' Top1 Accuracy {:.4f} Top5 Accuracy {:.4f}'.format(top1_score, top5_score))
            if DEBUG:
                break

    logger.debug('Training End')
    return top1_score, top5_score, mean_loss


def run_autoencoder(model_name, model, model_directory, number_of_epochs,
    learning_rate, logger, train_loader, val_loader, device, beta, image_size,
    gamma, image_directory=pathJoin('betavaeresults'), load_data=None,
    dataset_names=['stylized-imagenet200-0.0', 'stylized-imagenet200-1.0'], vae_transforms=None):
    checkpoint_path = pathJoin(model_directory, '{}.ckpt'.format(model_name))
    print(checkpoint_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_epoch = 0

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['weights'])
        optimizer.load_state_dict(checkpoint['optimizer_weights'])

    last_epoch += 1

    logger.info('Training model {} from epoch {}'.format(checkpoint_path, last_epoch))
    logger.info('Epochs {}'.format(number_of_epochs))
    logger.info('Batch Size {}'.format(train_loader.batch_size))
    logger.info('Number of Workers {}'.format(train_loader.num_workers))
    logger.info('Optimizer {}'.format(optimizer))
    logger.info('Learning Rate {}'.format(learning_rate))
    logger.info('Device {}'.format(device))

    image_directory = pathJoin(image_directory, model_name)
    os.makedirs(image_directory, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    distribution = 'bernoulli' if 'highpass' in model_name else 'gaussian'

    anneal_start = 30
    anneal_width = number_of_epochs - anneal_start

    max_beta = beta
    min_beta = 0.0

    current_beta = min_beta

    for epoch in range(last_epoch, number_of_epochs + 1):
        train_top1_accuracy, train_top5_accuracy, train_loss = train_autoencoder(
            model, train_loader, optimizer, logger, device, current_beta, gamma, criterion, distribution)

        reconstruction_grid_filename = pathJoin(image_directory, 'reconstructed_epoch_{}.png'.format(epoch))
        manifold_filename = pathJoin(image_directory, 'manifold_epoch_{}.png'.format(epoch))

        validation_top1_accuracy, validation_top5_accuracy, validation_loss = validate_autoencoder(
            model, val_loader, logger, device, reconstruction_grid_filename, manifold_filename, current_beta, gamma, criterion, ((epoch % 10) == 0), distribution)

        if epoch > anneal_start:
            current_beta += (max_beta - current_beta) / (anneal_width * 0.3)

        logger.info('Epoch {}: Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
            epoch, train_loss, train_top1_accuracy, train_top5_accuracy) \
            + ' Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
                validation_loss, validation_top1_accuracy, validation_top5_accuracy))

        if epoch % 10 == 0:
            with torch.no_grad():
                z = torch.randn(64, model.z_dim).to(device)
                sample = model._decode(z).cpu()
                sample_filename = pathJoin(image_directory, 'generated_epoch_{}.png'.format(epoch))
                save_image(sample.view(64, 3, image_size, image_size), sample_filename, normalize=False)

            # explore_betavae(model_name, model, image_directory, epoch, val_loader, device)

        logger.debug('Saving new weights')
        os.makedirs(model_directory, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'train_top1_accuracy': train_top1_accuracy,
            'train_top5_accuracy': train_top5_accuracy,
            'train_loss': train_loss,
            'validation_top1_accuracy': validation_top1_accuracy,
            'validation_top5_accuracy': validation_top5_accuracy,
            'validation_loss': validation_loss,
            'weights': model.state_dict(),
            'optimizer_weights': optimizer.state_dict()
        }
        torch.save(checkpoint, pathJoin(model_directory, '{}.ckpt'.format(model_name)))

    logger.info('Epoch {}'.format(checkpoint['epoch']))

    evaluate_model(model_name, model, load_data, dataset_names, logger.info, False, device, vae_transforms)
    logger.info('Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
        checkpoint['train_loss'], checkpoint['train_top1_accuracy'], checkpoint['train_top5_accuracy']))
    logger.info('Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
        checkpoint['validation_loss'], checkpoint['validation_top1_accuracy'], checkpoint['validation_top5_accuracy']))
    logger.info('Train: Loss: {:.4f}'.format(checkpoint['train_loss']))
    logger.info('Validation: Loss: {:.4f}'.format(checkpoint['validation_loss']))


# In[5]: Non-Training

def sanity(model_list, loader, pair_loader, device):
    for model_name in model_list:
        print(model_name)
        model = model_list[model_name]()
        model.train()
        dataloader = pair_loader if 'vae' in model_name else loader
        for batch in dataloader:
            index_image = dataloader.dataset.INDEX_IMAGE
            model(batch[index_image].to(device))
            break
        del model
        torch.cuda.empty_cache()


def perf(model_list, model_directory, dataset_names, device, load_data=None, load_bilateral_data=None, only_exists=None, vae_transforms=None):
    for model_name in model_list:
        print(model_name)
        model = model_list[model_name]()

        checkpoint_path = pathJoin(model_directory, '{}.ckpt'.format(model_name))
        print(checkpoint_path)
        
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)

            epoch = checkpoint['epoch']
            train_loss = checkpoint['train_loss']
            validation_loss = checkpoint['validation_loss']
            train_top1_accuracy = checkpoint['train_top1_accuracy'] if 'train_top1_accuracy' in checkpoint else 0.0
            train_top5_accuracy = checkpoint['train_top5_accuracy'] if 'train_top5_accuracy' in checkpoint else 0.0
            validation_top1_accuracy = checkpoint['validation_top1_accuracy'] if 'validation_top1_accuracy' in checkpoint else 0.0
            validation_top5_accuracy = checkpoint['validation_top5_accuracy'] if 'validation_top5_accuracy' in checkpoint else 0.0
            model.load_state_dict(checkpoint['weights'])

            print('Epoch: {} Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
                epoch, validation_loss, validation_top1_accuracy, validation_top5_accuracy) \
                + ' Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(
                    train_loss, train_top1_accuracy, train_top5_accuracy))

            if not only_exists:
                eval_transforms = vae_transforms if 'vae' in model_name else None
                evaluate_model(model_name, model, load_data, dataset_names, print, 'similarity' in model_name, device, eval_transforms)
                evaluate_model(model_name + '_eval_on_bilateral_images', model, load_bilateral_data, dataset_names, print, 'similarity' in model_name, device, eval_transforms)
        else:
            print('Checkpoint not available for model {}'.format(model_name))
        del model
        torch.cuda.empty_cache()

