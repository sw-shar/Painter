#!/usr/bin/env python3

import os
import time

import onnx

# from onnx2pytorch import ConvertModel
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from model import StyleTransferNetwork

IS_SAVING_ONNX = False
FILENAME_PTH = "modeldata/vgg_normalised.pth"
FILENAME_CHECKPOINT = "modeldata/StyleTransfer Checkpoint Iter_ 120000.tar"
FILENAME_ONNX = "modeldata/model.onnx"


def filename2mtime(filename):
    mtime = os.path.getmtime(filename)
    return time.ctime(mtime)


class Painter:
    def __init__(self):
        # state_vgg = ConvertModel(onnx.load('model.onnx'))
        state_vgg = torch.load(FILENAME_PTH, map_location=torch.device("cpu"))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.network = StyleTransferNetwork(
            self.device,
            state_vgg,
            training=False,
            load_fromstate=True,
            load_path=FILENAME_CHECKPOINT,
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(512),
                # transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )

        self.toPIL = transforms.ToPILImage(mode="RGB")

        try:
            self.model_onnx = onnx.load(FILENAME_ONNX)
        except FileNotFoundError:
            self.model_onnx = None

    def save_to_onnx(self, style, content, alpha):
        self.network.train(False)
        torch.onnx.export(self.network, (style, content, alpha), FILENAME_ONNX)
        model = onnx.load(FILENAME_ONNX)
        meta = model.metadata_props.add()
        meta.key = 'mtime'
        meta.value = filename2mtime(FILENAME_PTH)
        onnx.save(model, FILENAME_ONNX)
        return model

    def get_metadata(self):
        if self.model_onnx is None:  # todo
            return {}

        key2value = {
            prop.key: prop.value for prop in self.model_onnx.metadata_props
        }
        return {'mtime': key2value['mtime']}

    def paint(self, way_style, way_content, alpha, way_result=None):
        if way_result is None:
            way_result = way_content + '.result.jpg'

        # Загрузить изображение, преобразовать в RGB, преобразовать,
        # добавить размер 0 и переместить на устройство
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

        if self.model_onnx is None or IS_SAVING_ONNX:
            self.model_onnx = self.save_to_onnx(style, content, alpha)

        out, metrics = self.network(style, content, alpha)
        out = out.cpu()
        # convert to grid/image
        out = torchvision.utils.make_grid(
            out.clamp(min=-1, max=1), nrow=3, scale_each=True, normalize=True
        )
        # Make Pil
        img = self.toPIL(out)

        img.save(way_result)
        print('ok')
        return metrics


def main():
    painter = Painter()
    way_style = 'styleimages/abstraktsiya.jpg'
    way_content = 'tmp/moskva.jpg'
    alpha = 1

    print(painter.paint(way_style, way_content, alpha))
    print(painter.get_metadata())


if __name__ == '__main__':
    main()
