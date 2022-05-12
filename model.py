__all__ = ['StyleTransferNetwork']

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loss import find_metrics_loss
from util import calc_mean_std

# Декодер представляет собой перевернутый vgg19 до ReLU 4.1.
# Обратите внимание, что последний слой не активирован.


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.padding = nn.ReflectionPad2d(
            padding=1
        )  # Using reflection padding as described in vgg19
        self.UpSample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv4_1 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.conv3_1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv3_3 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv3_4 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.conv2_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.conv1_1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0
        )

    def forward(self, x):
        out = self.UpSample(F.relu(self.conv4_1(self.padding(x))))

        out = F.relu(self.conv3_1(self.padding(out)))
        out = F.relu(self.conv3_2(self.padding(out)))
        out = F.relu(self.conv3_3(self.padding(out)))
        out = self.UpSample(F.relu(self.conv3_4(self.padding(out))))

        out = F.relu(self.conv2_1(self.padding(out)))
        out = self.UpSample(F.relu(self.conv2_2(self.padding(out))))

        out = F.relu(self.conv1_1(self.padding(out)))
        out = self.conv1_2(self.padding(out))
        return out


# Последовательность vgg19, которая используется до Relu 4.1. Отметим, что
# первый слой представляет собой свертку 3,3, отличную от стандартной vgg19


class Encoder(nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        self.vgg19 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(
                inplace=True
            ),  # First layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(
                inplace=True
            ),  # Second layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(
                padding=1
            ),  # Third layer from which Style Loss is calculated
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(
                inplace=True
            ),  # This is Relu 4.1 The output layer of the encoder.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
        ).to(device)

        self.vgg19.load_state_dict(state_dict)

        encoder_children = list(self.vgg19.children())
        self.EncoderList = nn.ModuleList(
            [
                nn.Sequential(*encoder_children[:4]),  # Up to Relu 1.1
                nn.Sequential(*encoder_children[4:11]),  # Up to Relu 2.1
                nn.Sequential(*encoder_children[11:18]),  # Up to Relu 3.1
                nn.Sequential(
                    *encoder_children[18:31]
                ),  # Up to Relu 4.1, also the
            ]
        )  # input for the decoder

    def forward(
        self, x, intermediates=False
    ):  # if training use intermediates = True, to get the output of
        states = []  # all the encoder layers to calculate the style loss
        for i in range(len(self.EncoderList)):
            x = self.EncoderList[i](x)

            if intermediates:  # All intermediate states get saved in states
                states.append(x)
        if intermediates:
            return states
        return x


def AdaIn(content, style):
    # Только первые два затемнения, так что возможны разные размеры изображения
    assert content.shape[:2] == style.shape[:2]

    batch_size, n_channels = content.shape[:2]
    mean_content, std_content = calc_mean_std(content)
    mean_style, std_style = calc_mean_std(style)

    output = (
        std_style * ((content - mean_content) / (std_content)) + mean_style
    )  # Нормализуйте, затем измените среднее значение и станд.
    return output


class StyleTransferNetwork(nn.Module):
    def __init__(
        self,
        device,  # "cpu" for cpu, "cuda" for gpu
        enc_state_dict,  # Состояние предварительно обученного vgg19
        learning_rate=1e-4,
        learning_rate_decay=5e-5,  # Параметр затухания для скорости обучения
        gamma=6,
        # Управляет важностью StyleLoss и ContentLoss,
        # Loss = gamma*StyleLoss + ContentLoss
        training=True,  # Обучается сеть или нет
        load_fromstate=False,  # Загрузить с контрольной точки?
        load_path=None,  # Путь для загрузки контрольной точки
    ):
        super().__init__()

        assert device in ["cpu", "cuda"]
        if load_fromstate and not os.path.isfile(load_path):
            raise ValueError("Checkpoint file not found")

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.training = training
        self.gamma = gamma

        self.encoder = Encoder(enc_state_dict, device)
        # В качестве кодировщика используется предварительно обученный vgg19.

        self.decoder = Decoder().to(device)

        self.optimiser = optim.Adam(
            self.decoder.parameters(), lr=self.learning_rate
        )
        self.iters = 0

        if load_fromstate:
            state = torch.load(load_path, map_location=torch.device("cpu"))
            self.decoder.load_state_dict(state["Decoder"])
            self.optimiser.load_state_dict(state["Optimiser"])
            self.iters = state["iters"]

    def set_train(self, boolean):  # Изменить состояние сети
        assert type(boolean) == bool
        self.training = boolean

    def forward(self, style, content, alpha=1.0):
        # Альфа может использоваться во время тестирования
        # для контроля важности переданного стиля.

        # Encode style and content
        layers_style = self.encoder(
            style, self.training
        )  # if training: возвращает все состояния
        layer_content = self.encoder(
            content, False
        )  # для контента важен только последний слой

        # Transfer Style
        if self.training:
            style_applied = AdaIn(
                layer_content, layers_style[-1]
            )  # Последний слой - это слой «стиль».
        else:
            style_applied = (
                alpha * AdaIn(layer_content, layers_style)
                + (1 - alpha) * layer_content
            )  # Альфа контролирует величину стиля

        # Scale up
        style_applied_upscaled = self.decoder(style_applied)

        # Вычислить потери
        layers_style_applied = self.encoder(
            style_applied_upscaled, self.training
        )

        metrics = find_metrics_loss(
            layers_style_applied, layer_content, layers_style, self.gamma
        )
        return style_applied_upscaled, metrics

