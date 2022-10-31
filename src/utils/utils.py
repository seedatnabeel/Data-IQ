# stdlib
import logging

# third party
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainData(Dataset):
    # Conventional train dataset
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TrainDataDRO(Dataset):
    # DRO dataset
    def __init__(self, X_data, y_data, y_sub):
        self.X_data = X_data
        self.y_data = y_data
        self.y_sub = y_sub

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.y_sub[index]

    def __len__(self):
        return len(self.X_data)


# Adapted from:
# https://github.com/kohpangwei/group_DRO &
# https://github.com/HazyResearch/hidden-stratification
class LossComputer:
    # Loss computer for Group-DRO
    def __init__(
        self,
        criterion,
        is_robust,
        n_groups,
        group_counts,
        robust_step_size,
        stable=True,
        size_adjustments=None,
        auroc_version=False,
        class_map=None,
        use_cuda=True,
    ):
        self.criterion = criterion
        self.is_robust = is_robust
        self.auroc_version = auroc_version
        self.n_groups = n_groups
        if auroc_version:
            assert class_map is not None
            self.n_gdro_groups = len(class_map[0]) * len(class_map[1])
            self.class_map = class_map
        else:
            self.n_gdro_groups = n_groups
        self.group_range = torch.arange(self.n_groups).unsqueeze(1).long()
        if use_cuda:
            self.group_range = self.group_range.cuda()

        if self.is_robust:
            self.robust_step_size = robust_step_size
            logging.info(
                f"Using robust loss with inner step size {self.robust_step_size}",
            )
            self.stable = stable
            self.group_counts = group_counts.to(self.group_range.device)

            if size_adjustments is not None:
                self.do_adj = True
                if auroc_version:
                    self.adj = (
                        torch.tensor(size_adjustments[0])
                        .float()
                        .to(self.group_range.device)
                    )
                    self.loss_adjustment = self.adj / torch.sqrt(self.group_counts[:-1])
                else:
                    self.adj = (
                        torch.tensor(size_adjustments)
                        .float()
                        .to(self.group_range.device)
                    )
                    self.loss_adjustment = self.adj / torch.sqrt(self.group_counts)
            else:
                self.adj = (
                    torch.zeros(self.n_gdro_groups).float().to(self.group_range.device)
                )
                self.do_adj = False
                self.loss_adjustment = self.adj

            logging.info(
                f"Per-group loss adjustments: {np.round(self.loss_adjustment.tolist(), 2)}",
            )
            # The following quantities are maintained/updated throughout training
            if self.stable:
                logging.info("Using numerically stabilized DRO algorithm")
                self.adv_probs_logits = torch.zeros(self.n_gdro_groups).to(
                    self.group_range.device,
                )
            else:  # for debugging purposes
                logging.warn("Using original DRO algorithm")
                self.adv_probs = (
                    torch.ones(self.n_gdro_groups).to(self.group_range.device)
                    / self.n_gdro_groups
                )
        else:
            logging.info("Using ERM")

    def loss(self, yhat, y, group_idx=None, is_training=False):
        """
        The function takes in the predicted values, the actual values, and the group index. It then
        computes the per-sample and per-group losses. It then computes the overall loss.

        Args:
          yhat: the output of the model
          y: the true labels
          group_idx: the index of the group that each sample belongs to.
          is_training: whether the model is in training mode or not. Defaults to False
        """
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        y.shape[0]

        group_losses, group_counts = self.compute_group_avg(
            per_sample_losses,
            group_idx,
        )
        corrects = (torch.argmax(yhat, 1) == y).float()
        group_accs, group_counts = self.compute_group_avg(corrects, group_idx)

        # compute overall loss
        if self.is_robust:
            if self.auroc_version:
                neg_subclasses, pos_subclasses = self.class_map[0], self.class_map[1]
                pair_losses = []
                for neg_subclass in neg_subclasses:
                    neg_count = group_counts[neg_subclass]
                    neg_sbc_loss = group_losses[neg_subclass] * neg_count
                    for pos_subclass in pos_subclasses:
                        pos_count = group_counts[pos_subclass]
                        pos_sbc_loss = group_losses[pos_subclass] * pos_count
                        tot_count = neg_count + pos_count
                        tot_count = tot_count + (tot_count == 0).float()
                        pair_loss = (neg_sbc_loss + pos_sbc_loss) / tot_count
                        pair_losses.append(pair_loss)
                loss, _ = self.compute_robust_loss(
                    torch.cat([pl.view(1) for pl in pair_losses]),
                )
            else:
                loss, _ = self.compute_robust_loss(group_losses)
        else:
            loss = per_sample_losses.mean()

        return (
            loss,
            (per_sample_losses, corrects),
            (group_losses, group_accs, group_counts),
        )

    def compute_robust_loss(self, group_loss):
        """
        The function takes in the per-group losses and computes the robust loss.
        Args:
          group_loss: the loss of each group

        Returns:
          The robust loss and the adjusted probabilities.
        """
        if torch.is_grad_enabled():  # update adv_probs if in training mode
            adjusted_loss = group_loss
            if self.do_adj:
                adjusted_loss += self.loss_adjustment
            logit_step = self.robust_step_size * adjusted_loss.data
            if self.stable:
                self.adv_probs_logits = self.adv_probs_logits + logit_step
            else:
                self.adv_probs = self.adv_probs * torch.exp(logit_step)
                self.adv_probs = self.adv_probs / self.adv_probs.sum()

        if self.stable:
            adv_probs = torch.softmax(self.adv_probs_logits, dim=-1)
        else:
            adv_probs = self.adv_probs
        robust_loss = group_loss @ adv_probs
        return robust_loss, adv_probs

    def compute_group_avg(self, losses, group_idx, num_groups=None, reweight=None):
        """
        It takes in a list of losses, a list of group indices, and a list of reweights, and returns a
        list of group losses and a list of group counts

        Args:
          losses: the losses of the model
          group_idx: the group index of each sample
          num_groups: the number of groups to split the data into.
          reweight: if True, reweight the loss by the inverse of the group size

        Returns:
          group_loss, group_count
        """
        # compute observed counts and mean loss for each group
        if num_groups is None:
            group_range = self.group_range
        else:
            group_range = (
                torch.arange(num_groups).unsqueeze(1).long().to(group_idx.device)
            )
        # reweight=True
        # num_groups=2
        if reweight is not None:
            group_loss, group_count = [], []
            reweighted = losses * reweight
            for i in range(num_groups):
                inds = group_idx == i
                group_losses = reweighted[inds]
                group_denom = torch.sum(reweight[inds])
                group_denom = group_denom
                group_loss.append(
                    torch.sum(group_losses)
                    / (group_denom + (group_denom == 0).float()),
                )
                group_count.append(group_denom)
            group_loss, group_count = torch.tensor(group_loss), torch.tensor(
                group_count,
            )
        else:
            group_map = (group_idx == group_range).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count == 0).float()  # avoid nans
            group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def __call__(self, yhat, y, group_idx):
        return self.loss(yhat, y, group_idx)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, delta=0.0001, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""

        print(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...",
        )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_scores_generic(overall_list, combos):
    """
    Return the generic correlations

    Args:
      overall_list: a dictionary of lists, where each list is a list of scores for a particular metric.
      combos: a list of tuples, each tuple is a pair of indices of the overall_list

    Returns:
      The mean of the pearson and spearman correlation
    """
    # third party
    from scipy import stats

    corr = []  # pearson
    corr2 = []  # spearman
    for combo in combos:
        rvs1 = overall_list[combo[0]]
        rvs2 = overall_list[combo[1]]
        corr.append(np.corrcoef(rvs1, rvs2)[0, 1])
        corr2.append(stats.spearmanr(rvs1, rvs2)[0])

    return np.mean(corr), np.mean(corr2)


def get_grad_scores(dataiqs, combos):
    """
    Returns GradN correlation scores

    Args:
      dataiqs: a dictionary of dataiqs, where the keys are the names of the datamaps
      combos: list of tuples of strings, each tuple is a pair of conditions to compare

    Returns:
      The mean of the pearson and spearman correlations
    """
    # third party
    from scipy import stats

    corr = []  # pearson
    corr2 = []  # spearman
    for combo in combos:
        rvs1 = np.mean(dataiqs[combo[0]].get_grads, axis=1)
        rvs2 = np.mean(dataiqs[combo[1]].get_grads, axis=1)
        corr.append(np.corrcoef(rvs1, rvs2)[0, 1])
        corr2.append(stats.spearmanr(rvs1, rvs2)[0])

    return np.mean(corr), np.mean(corr2)


def get_dataiq_scores(dataiqs, combos, feat):
    """
    Returns the average correlation of metrics for Data-IQ or Data Maps
    Args:
      dataiqs: a dictionary of DataIQ objects
      combos: list of tuples of model names
      feat: "variability" or "aleatoric"

    Returns:
      the average of the pearson and spearman correlation
    """
    # third party
    from scipy import stats

    corr = []  # pearson
    corr2 = []  # spearman

    # uncertainty score
    for combo in combos:
        if feat == "variability":
            rvs1 = dataiqs[combo[0]].variability
            rvs2 = dataiqs[combo[1]].variability
        else:
            rvs1 = dataiqs[combo[0]].aleatoric
            rvs2 = dataiqs[combo[1]].aleatoric

        corr.append(np.corrcoef(rvs1, rvs2)[0, 1])
        corr2.append(stats.spearmanr(rvs1, rvs2)[0])

    varx1 = np.mean(corr)
    varx2 = np.mean(corr2)

    corr = []  # pearson
    corr2 = []  # spearman

    # confidence
    for combo in combos:
        rvs1 = dataiqs[combo[0]].confidence
        rvs2 = dataiqs[combo[1]].confidence
        corr.append(np.corrcoef(rvs1, rvs2)[0, 1])
        corr2.append(stats.spearmanr(rvs1, rvs2)[0])

    vary1 = np.mean(corr)
    vary2 = np.mean(corr2)

    r1 = (varx1 + vary1) / 2
    r2 = (varx2 + vary2) / 2
    return r1, r2


def compare_model_classes(model1, model2, feat=None):
    # third party
    from scipy import stats

    if feat == "variability":
        rvs1 = model1.variability
        rvs2 = model2.variability
    else:
        rvs1 = model1.aleatoric
        rvs2 = model2.aleatoric

    x_corr1 = np.corrcoef(rvs1, rvs2)[0, 1]  # pearson
    x_corr2 = stats.spearmanr(rvs1, rvs2)[0]  # spearman

    rvs1 = model1.confidence
    rvs2 = model2.confidence

    y_corr1 = np.corrcoef(rvs1, rvs2)[0, 1]  # pearson
    y_corr2 = stats.spearmanr(rvs1, rvs2)[0]  # spearman

    r1 = (x_corr1 + y_corr1) / 2
    r2 = (x_corr2 + y_corr2) / 2
    return x_corr1, x_corr2, y_corr1, y_corr2, r1, r2


def compute_jtt(X_train, y_train, net):
    """
    Computes the JTT scores (i.e. where the model is right or wrong)

    Args:
      X_train: the training data
      y_train: the labels of the training data
      net: the network you trained

    Returns:
      list of JTT scores
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_pred_list = []
    net_test = net
    net_test.eval()
    with torch.no_grad():
        X_batch = torch.tensor(X_train)
        X_batch = X_batch.to(device)
        y_train_pred = net_test(X_batch)
        _, predicted = torch.max(y_train_pred.data, 1)
        y_pred_list.append(predicted.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list][0]
    return (y_pred_list == y_train).astype(int)
