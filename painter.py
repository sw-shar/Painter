import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import img_as_float, io
from torchvision.utils import save_image


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


class StyleTransferNetwork(nn.Module):
    def __init__(
        self,
        device,  # "cpu" for cpu, "cuda" for gpu
        enc_state_dict,  # Состояние предварительно обученного vgg19
        learning_rate=1e-4,
        learning_rate_decay=5e-5,  # Параметр затухания для скорости обучения
        gamma=6,  # Управляет важностью StyleLoss и ContentLoss, Loss = gamma*StyleLoss + ContentLoss
        train=True,  # Обучается сеть или нет
        load_fromstate=False,  # Загрузить с контрольной точки?
        load_path=None,  # Путь для загрузки контрольной точки
    ):
        super().__init__()

        assert device in ["cpu", "cuda"]
        if load_fromstate and not os.path.isfile(load_path):
            raise ValueError("Checkpoint file not found")

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.train = train
        self.gamma = gamma

        self.encoder = Encoder(
            enc_state_dict, device
        )  # В качестве кодировщика используется предварительно обученный vgg19.
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
        self.train = boolean

    def adjust_learning_rate(
        self, optimiser, iters
    ):  # Простое снижение скорости обучения
        lr = learning_rate / (1.0 + learning_rate_decay * iters)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

    def forward(
        self, style, content, alpha=1.0
    ):  # Альфа может использоваться во время тестирования для контроля важности переданного стиля.

        # Encode style and content
        layers_style = self.encoder(
            style, self.train
        )  # if train: возвращает все состояния
        layer_content = self.encoder(
            content, False
        )  # для контента важен только последний слой

        # Transfer Style
        if self.train:
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
        if not self.train:
            return style_applied_upscaled  # Когда не тренируется, возвращайте преобразованное изображение

        # Вычислить потери
        layers_style_applied = self.encoder(style_applied_upscaled, self.train)

        content_loss = Content_loss(layers_style_applied[-1], layer_content)
        style_loss = Style_loss(layers_style_applied, layers_style)

        loss_comb = content_loss + self.gamma * style_loss

        return loss_comb, content_loss, style_loss


# Декодер представляет собой перевернутый vgg19 до ReLU 4.1. Обратите внимание, что последний слой не активирован.


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


def AdaIn(content, style):
    assert (
        content.shape[:2] == style.shape[:2]
    )  # Только первые два затемнения, так что возможны разные размеры изображения
    batch_size, n_channels = content.shape[:2]
    mean_content, std_content = calc_mean_std(content)
    mean_style, std_style = calc_mean_std(style)

    output = (
        std_style * ((content - mean_content) / (std_content)) + mean_style
    )  # Нормализуйте, затем измените среднее значение и станд.
    return output


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


def is_jupyter():
    try:
        __file__
    except:
        return True
    return False


class Painter:
    def __init__(self):
        state_vgg = torch.load(
            "modeldata/vgg_normalised.pth", map_location=torch.device("cpu")
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.network = StyleTransferNetwork(
            self.device,
            state_vgg,
            train=False,
            load_fromstate=True,
            load_path="modeldata/StyleTransfer Checkpoint Iter_ 120000.tar",
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(512),
                # transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )

        self.toPIL = transforms.ToPILImage(mode="RGB")

    def paint(self, way_style, way_content, alpha, way_result=None):
        if way_result is None:
            way_result = way_content + '.result.jpg'

        # Загрузить изображение, преобразовать в RGB, преобразовать, добавить размер 0 и переместить на устройство
        style = (
            self.transform(Image.open(way_style).convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )
        content = (
            self.transform(Image.open(way_content).convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )

        style_img = img_as_float(io.imread(way_style))
        content_img = img_as_float(io.imread(way_content))

        out = self.network(style, content, alpha).cpu()
        # convert to grid/image
        out = torchvision.utils.make_grid(
            out.clamp(min=-1, max=1), nrow=3, scale_each=True, normalize=True
        )
        # Make Pil
        img = self.toPIL(out)

        if not is_jupyter():
            img.save(way_result)
            print('ok')
            return

        images = [style_img, content_img, img]
        titles = ['Style', 'Content', 'Adaptive']

        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 20))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(images[i])
            plt.title(titles[i])


def main():
    painter = Painter()
    way_style = 'media/abstraktsiya.jpg'
    way_content = "media/moskva.jpg"
    alpha = 1

    painter.paint(way_style, way_content, alpha)


if __name__ == '__main__':
    main()
