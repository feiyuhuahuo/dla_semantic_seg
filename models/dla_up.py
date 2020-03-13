import math
import numpy as np
import torch
from torch import nn
from models import dla


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super().__init__()
        self.channels = channels
        self.out_dim = out_dim
        self.nested = False

        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(nn.Conv2d(c, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                                        output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            C_in = out_dim * (i + 1) if self.nested else out_dim * 2
            ##############################################################
            # if i >= 2:
            #     node = nn.Sequential(DCN(C_in, out_dim, kernel_size=node_kernel, stride=1,
            #                              padding=node_kernel // 2, deformable_groups=1),
            #                          nn.BatchNorm2d(out_dim),
            #                          nn.ReLU(inplace=True))
            ##############################################################
            # else:
            node = nn.Sequential(nn.Conv2d(C_in, out_dim, kernel_size=node_kernel, stride=1,
                                           padding=node_kernel // 2, bias=False),
                                 nn.BatchNorm2d(out_dim),
                                 nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), f'{len(self.channels)} vs {len(layers)} layers'
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))

        x = layers[0]
        y = []

        skip_cat = [x]
        for i in range(1, len(layers)):
            node = getattr(self, f'node_{i}')

            if i >= 2 and self.nested:  # Nested IDAup implementation.
                additional_in = skip_cat[:i - 1]
                additional_in.reverse()
                x = node(torch.cat([x, layers[i]] + additional_in, 1))
            else:
                x = node(torch.cat((x, layers[i]), 1))

            skip_cat.append(x)
            y.append(x)

        return x, y


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super().__init__()
        if in_channels is None:
            in_channels = channels

        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, f'ida_{i}', IDAUp(3, channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1

        for i in range(len(layers) - 1):
            ida = getattr(self, f'ida_{i}')
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y

        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, classes, down_ratio=2):
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.base = dla.__dict__[base_name]()
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)  # [32, 64, 128, 256, 512], [1, 2, 4, 8, 16]
        self.fc = nn.Sequential(nn.Conv2d(channels[self.first_level], classes, kernel_size=3, stride=1, padding=1))

        up_factor = 2 ** self.first_level
        if up_factor > 1:
            up = nn.ConvTranspose2d(classes, classes, up_factor * 2, stride=up_factor, padding=up_factor // 2,
                                    output_padding=0, groups=classes, bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
        else:
            up = Identity()
        self.up = up

        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        x = self.fc(x)
        x = self.up(x)

        return x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.dla_up.parameters():
            yield param
        for param in self.fc.parameters():
            yield param
