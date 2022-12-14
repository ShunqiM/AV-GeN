import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    def __init__(self, cri) -> None:
        super(ContrastiveLoss, self).__init__()
        self.cri = cri

    def __str__(self) -> str:
        return "Contrast Loss"

    def forward(self, pred, target):
        loss = self.cri(pred, target) + torch.abs((pred[:, 0, :, :] - pred[:, 1, :, :]) - (target[:, 0, :, :] - target[:, 1, :, :])).mean()
        return loss

class InfoNCELoss(nn.Module):
    def __init__(self, temp = 0.07) -> None:
        super(InfoNCELoss, self).__init__()
        self.cri = torch.nn.CrossEntropyLoss().cuda()
        self.temp = temp

    def __str__(self) -> str:
        return "Info NCE Loss"

    def forward(self, features, alternative_features):
        features = torch.cat([features, alternative_features], dim=0)
        labels = torch.cat([torch.arange(features.shape[0]//2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temp
        loss = self.cri(logits, labels)
        return loss

def info_nce_loss(features, alternative_features):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    features = torch.cat([features, alternative_features], dim=0)
    labels = torch.cat([torch.arange(features.shape[0]//2) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / 0.07
    loss = criterion(logits, labels)
    return loss

class KDLoss(nn.Module):
    def __init__(self, tmp = 7) -> None:
        super(KDLoss, self).__init__()
        self.cri = nn.KLDivLoss(reduction="batchmean")
        self.tmp = tmp

    def __str__(self) -> str:
        return "KD Loss"

    def forward(self, pred, target):
        ditillation_loss = self.cri(
            F.softmax(pred / self.tmp, dim=1),
            F.softmax(target / self.tmp, dim=1)
        )
        return ditillation_loss

class NCLoss(nn.Module):
    def __init__(self) -> None:
        super(NCLoss, self).__init__()

    def __str__(self) -> str:
        return "Normalised Contrast Loss"

    def forward(self, pred, target):
        l1_err = torch.abs(torch.mean((target - pred), dim=2))
        c_err = torch.abs((pred[:, 0, :, :] - pred[:, 1, :, :]) - (target[:, 0, :, :] - target[:, 1, :, :]))
        c_err = torch.mean(c_err, dim = 2)
        loss = l1_err + c_err
        loss /= target.mean(dim=2)
        return loss

class SpectraLoss(nn.Module):
    # PARALLEL WAVEGAN: A FAST WAVEFORM GENERATION MODEL BASED ON GENERATIVE ADVERSARIAL NETWORKS WITH MULTI-RESOLUTION SPECTROGRAM
    def __init__(self) -> None:
        super(SpectraLoss, self).__init__()
        self.l2 = nn.MSELoss()

    def __str__(self) -> str:
        return "SpectraLoss"

    def forward(self, pred, target):
        err = torch.linalg.norm(target - pred) / torch.linalg.norm(target)
        loss = self.l2(torch.log1p(pred), torch.log1p(target)) + err
        return loss


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight)
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err

class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))