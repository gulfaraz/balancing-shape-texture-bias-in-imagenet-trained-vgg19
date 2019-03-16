import torch
import numpy as np
from score import *

def validate(model, dataloader, criterion, logger, device):
    logger.debug('Validation Start')
    model.eval()
    
    total_top1, total_top5, total_, top1_score, top5_score = 0, 0, 0, 0, 0
    loss = []

    for batch_index, batch in enumerate(dataloader):
        output = model(batch[dataloader.dataset.INDEX_IMAGE].to(device))
        target = batch[dataloader.dataset.INDEX_TARGET].to(device)

        _, predicted_class = output.topk(5, 1, True, True)
        top1, top5, total = score(predicted_class, target)

        total_top1 += top1
        total_top5 += top5
        total_ += total

        # loss
        batch_loss = criterion(output, target)
        
        loss.append(batch_loss.item())

        mean_loss = np.mean(loss)

        top1_score = score_value(total_top1, total_)
        top5_score = score_value(total_top5, total_)
        if (batch_index + 1) % 10 == 0:
            logger.debug('Validation Batch {0}/{1}: Top1 Accuracy {2:.4f} Top5 Accuracy {3:.4f} Loss {4:.4f}'.format(batch_index + 1, len(dataloader), top1_score, top5_score, mean_loss))

    logger.debug('Validation End')
    return top1_score, top5_score, mean_loss


def train(model, dataloader, criterion, optimizer, logger, device, grad_clip_norm_value=50):
    logger.debug('Training Start')
    model.train()

    total_top1, total_top5, total_, top1_score, top5_score = 0, 0, 0, 0, 0
    loss = []

    for batch_index, batch in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(batch[dataloader.dataset.INDEX_IMAGE].to(device))
        target = batch[dataloader.dataset.INDEX_TARGET].to(device)

        # accuracy
        _, predicted_class = output.topk(5, 1, True, True)
        top1, top5, total = score(predicted_class, target)

        total_top1 += top1
        total_top5 += top5
        total_ += total

        # loss
        batch_loss = criterion(output, target)
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
            logger.debug('Training Batch {0}/{1}: Top1 Accuracy {2:.4f} Top5 Accuracy {3:.4f} Loss {4:.4f}'.format(batch_index + 1, len(dataloader), top1_score, top5_score, mean_loss))

    logger.debug('Training End')
    return top1_score, top5_score, mean_loss


def run(run_name, model, training, number_of_epochs, logger, train_loader, val_loader, device):

    criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, min_lr=1e-5, verbose=True)
    
    best_validation_accuracy = -1.0

    for epoch in range(1, number_of_epochs + 1):
        if training:
            train_top1_accuracy, train_top5_accuracy, train_loss = train(model, train_loader, criterion, optimizer, logger, device)
        validation_top1_accuracy, validation_top5_accuracy, validation_loss = validate(model, val_loader, criterion, logger, device)
        logger.info('Epoch {0}: Train: Loss: {1:.4f} Top1 Accuracy: {2:.4f} Top5 Accuracy: {3:.4f} Validation: Loss: {4:.4f} Top1 Accuracy: {5:.4f} Top5 Accuracy: {6:.4f}'.format(epoch, train_loss, train_top1_accuracy, train_top5_accuracy, validation_loss, validation_top1_accuracy, validation_top5_accuracy))

        lr_scheduler.step(validation_loss)

        if validation_top5_accuracy > best_validation_accuracy:
            logger.debug('Improved Validation Score, saving new weights')
            model_directory = pathJoin(ROOT_PATH, 'models')
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
            torch.save(checkpoint, pathJoin(model_directory, '{}.ckpt'.format(run_name)))
            best_validation_accuracy = validation_top5_accuracy

    logger.info('Epoch {}'.format(checkpoint['epoch']))

    evaluate_model(model, logger.info)

    logger.info('Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(checkpoint['train_loss'], checkpoint['train_top1_accuracy'], checkpoint['train_top5_accuracy']))
    logger.info('Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(checkpoint['validation_loss'], checkpoint['validation_top1_accuracy'], checkpoint['validation_top5_accuracy']))

