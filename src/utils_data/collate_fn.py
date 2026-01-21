import torch

def my_collate_fn(batch):
    images = []
    labels = []
    for img, lbl in batch:
        images.append(img)
        labels.append(lbl)
    return torch.stack(images), torch.tensor(labels)
