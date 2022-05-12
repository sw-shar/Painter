import torch


# Вычисляет среднее и стандартное значение по каналам
def calc_mean_std(input, eps=1e-5):
    batch_size, channels = input.shape[:2]

    reshaped = input.view(batch_size, channels, -1)  # Изменить форму канала
    mean = torch.mean(reshaped, dim=2).view(
        batch_size, channels, 1, 1
    )  # Вычислить среднее значение и изменить форму
    std = torch.sqrt(torch.var(reshaped, dim=2) + eps).view(
        batch_size, channels, 1, 1
    )  # Вычислите дисперсию, добавьте эпсилон (избегайте деления на 0),
    # рассчитать std и изменить форму
    return mean, std

