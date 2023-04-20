import torch.nn.functional as F
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss
from itertools import product


class RankLoss:

    @staticmethod
    def softmax_ce_loss(y_pred, *args, **kwargs):
        return F.cross_entropy(y_pred, torch.zeros((y_pred.size(0),)).long().cuda())

    @staticmethod
    def pointwise_rmse(y_pred, y_true=None):
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1
        errors = (y_true - y_pred)
        squared_errors = errors ** 2
        valid_mask = (y_true != -100).float()
        mean_squared_errors = torch.sum(squared_errors, dim=1) / torch.sum(valid_mask, dim=1)
        rmses = torch.sqrt(mean_squared_errors)
        return torch.mean(rmses)

    @staticmethod
    def pointwise_bce(y_pred, y_true=None):
        if y_true is None:
            y_true = torch.zeros_like(y_pred).float().to(y_pred.device)
            y_true[:, 0] = 1
        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y_true)
        return loss

    @staticmethod
    def list_net(y_pred, y_true=None, padded_value_indicator=-100, eps=1e-10):
        """
            ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
            :param y_pred: predictions from the model, shape [batch_size, slate_length]
            :param y_true: ground truth labels, shape [batch_size, slate_length]
            :param eps: epsilon value, used for numerical stability
            :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
            :return: loss value, a torch.Tensor
            """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))

    @staticmethod
    def rank_net(y_pred, y_true=None, padded_value_indicator=-100, weight_by_diff=False,
                 weight_by_diff_powed=False):
        """
        RankNet loss introduced in "Learning to Rank using Gradient Descent".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
        :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
        :return: loss value, a torch.Tensor
        """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        # here we generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        # here we filter just the pairs that are 'positive' and did not involve a padded instance
        # we can do that since in the candidate pairs we had symetric pairs so we can stick with
        # positive ones for a simpler loss function formulation
        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

        pred_diffs = pred_diffs[the_mask]

        weight = None
        if weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

        # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
        # whether one document is better than the other and not about the actual difference in
        # their relevancy levels
        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)

    @staticmethod
    def lambda_loss(y_pred, y_true=None, eps=1e-10, padded_value_indicator=-100, weighing_scheme=None, k=None,
                    sigma=1., mu=10., reduction="mean", reduction_log="binary"):
        """
        LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
        Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
        :param k: rank at which the loss is truncated
        :param sigma: score difference weight used in the sigmoid function
        :param mu: optional weight used in NDCGLoss2++ weighing scheme
        :param reduction: losses reduction method, could be either a sum or a mean
        :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
        :return: loss value, a torch.Tensor
        """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        device = y_pred.device

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if weighing_scheme is None:
            weights = 1.
        else:
            weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        if reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss


def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()

    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
        G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args):
    return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args):
    return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args):
    return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))
