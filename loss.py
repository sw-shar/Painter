__all__ = ['find_metrics_loss']

import torch.nn.functional as F

from util import calc_mean_std


def Content_loss(input, target):  # Потеря контента — это простая потеря MSE
    loss = F.mse_loss(input, target)
    return loss


def Style_loss(input, target):
    mean_loss, std_loss = 0, 0

    for input_layer, target_layer in zip(input, target):
        mean_input_layer, std_input_layer = calc_mean_std(input_layer)
        mean_target_layer, std_target_layer = calc_mean_std(target_layer)

        mean_loss += F.mse_loss(mean_input_layer, mean_target_layer)
        std_loss += F.mse_loss(std_input_layer, std_target_layer)

    return mean_loss + std_loss


def find_metrics_loss(
    layers_style_applied, layer_content, layers_style, gamma
):
    content_loss = Content_loss(layers_style_applied[-1], layer_content)
    style_loss = Style_loss(layers_style_applied, layers_style)

    loss_comb = content_loss + gamma * style_loss

    return {
        'content_loss': float(content_loss),
        'style_loss': float(style_loss),
        'loss_comb': float(loss_comb),
    }

