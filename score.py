import torch
from tqdm import tqdm

def score(prediction, target):
    total = prediction.size(0)
    prediction = prediction.t()
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))
    top1 = correct[:1].view(-1).float().sum(0).item()
    top5 = correct[:5].view(-1).float().sum(0).item()
    return top1, top5, total


def score_value(score, total):
    if total > 0:
        return score/total
    else:
        return 0


def score_model(model, dataloader, device, similarity_model=False):
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_ = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            target = batch[dataloader.dataset.INDEX_TARGET].to(device)
            input = batch[dataloader.dataset.INDEX_IMAGE].to(device)
            if similarity_model:
                output, _ = model(input)
            else:
                output = model(input)
            _, predicted_classes = output.topk(5, 1, True, True)
            top1, top5, total = score(predicted_classes, target)
            total_top1 += top1
            total_top5 += top5
            total_ += total
    return total_top1/total_, total_top5/total_


def evaluate_model(model, load_data, dataset_names, print_function, similarity_model, device):
    scores = {
        'top5': [],
        'top1': []
    }

    for dataset_name in dataset_names:
        _, loader = load_data(dataset_name, split='val')
        top1, top5 = score_model(model, loader, device, similarity_model)
        print_function('{}: Top1: {:.4f} Top5: {:.4f}'.format(dataset_name, top1, top5))
        scores['top5'].append(top5)
        scores['top1'].append(top1)

    for metric in scores:
        print('{} => {}'.format(metric, ', '.join([ '{:.4f}'.format(x) for x in scores[metric] ])))

