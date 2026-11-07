import torch
import torch.nn as nn
from Method_Dataset_Loader import Dataset_CIFAR
from Method_CNN_Base import Method_CNN_Base


def conv_block(in_c, out_c):
    # two 3x3 convs, then maxpool halves height and width
    return [
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
    ]


class Method_CNN_CIFAR(Method_CNN_Base):
    learning_rate = 0.001
    batch_size = 64
    max_epoch = 30
    weight_decay = 1e-4

    def __init__(self, in_channels=3, image_size=(32, 32), num_classes=10):
        super().__init__()
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_layers = nn.Sequential(
            *conv_block(in_channels, 32),
            *conv_block(32, 64),
            *conv_block(64, 128),
        )
        flat = self.flat_dim(self.conv_layers.eval(), (in_channels, *image_size))
        self.conv_layers.train()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),  # 10 object classes in cifar-10
        )


if __name__ == '__main__':
    torch.manual_seed(1)
    train_set = Dataset_CIFAR('train')
    test_set = Dataset_CIFAR('test')

    model = Method_CNN_CIFAR()
    model.train_model(train_set)

    scores = model.evaluate(test_set)
    print('Accuracy:', scores['accuracy'])
    print('Precision:', scores['precision'])
    print('Recall:', scores['recall'])
    print('F1:', scores['f1'])

    path = model.plot_learning_curve('CIFAR_learning_curve.png', 'CNN Training Loss (CIFAR)')
    print('saved', path)