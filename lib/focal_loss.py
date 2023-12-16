class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=3):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.ones(num_classes) * alpha
            else:
                self.alpha = torch.tensor(alpha)
        self.alpha = self.alpha / self.alpha.sum()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).to(inputs.device)

        # Compute the log softmax
        log_softmax = F.log_softmax(inputs, dim=1)

        # Compute the loss per class
        loss_per_class = -targets_one_hot * log_softmax

        # Compute the focal loss factors
        softmax_probs = torch.exp(log_softmax)
        focal_factors = (1 - softmax_probs) ** self.gamma

        # Apply alpha weighting and focal factors
        alpha_factors = self.alpha.to(inputs.device).unsqueeze(0)
        loss = alpha_factors * focal_factors * loss_per_class

        # Sum over classes and compute the final loss based on reduction
        loss = loss.sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss