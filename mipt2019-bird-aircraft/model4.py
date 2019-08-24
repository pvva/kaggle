import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
import torch
import torch.nn as nn
import random
from PIL import Image


# Competition: https://www.kaggle.com/c/mipt2019-bird-aircraft
# This model gives score of 0.93


def convert_strings_to_categories_codes(df):
    for n, c in df.items():
        if is_string_dtype(c):
            df[n] = c.astype("category").cat.codes


PATH = "C:/Dev/data/mipt2019-bird-aircraft/"
train_x = pd.read_csv(f"{PATH}train_x.csv", index_col=0, header=None)
train_y = pd.read_csv(f"{PATH}train_y.csv", index_col=0)
test_x = pd.read_csv(f"{PATH}test_x.csv", index_col=0, header=None)

convert_strings_to_categories_codes(train_y)
# without augmentation the score is around 0.915
# images_train = (
#     train_x.values.reshape(train_x.shape[0], 32, 32, 3).swapaxes(1, 3).swapaxes(2, 3)
# )
#
# images_test = (
#     test_x.values.reshape(test_x.shape[0], 32, 32, 3).swapaxes(1, 3).swapaxes(2, 3)
# )
# train_y = train_y["target"].to_numpy().astype(np.int64)

# with augmentation
img_t = train_x.values.reshape(train_x.shape[0], 32, 32, 3)
img_aug = np.empty(img_t.shape, dtype=int)

for idx in range(img_t.shape[0]):
    pilImg = Image.fromarray(np.uint8(img_t[idx]))
    pilImg = pilImg.rotate(random.uniform(-15, 15), Image.BILINEAR)
    if random.random() < 0.5:
        pilImg = pilImg.transpose(Image.FLIP_LEFT_RIGHT)

    img_aug[idx] = np.asarray(pilImg)

# adding permutation for augmented training data increases score up to 0.935 (from 0.93)
aug_indices = np.random.permutation(img_aug.shape[0])
img_aug = img_aug[aug_indices]

img_t = np.concatenate((img_t, img_aug))
images_train = img_t.swapaxes(1, 3).swapaxes(2, 3)

images_test = (
    test_x.values.reshape(test_x.shape[0], 32, 32, 3).swapaxes(1, 3).swapaxes(2, 3)
)

train_y = train_y["target"].to_numpy().astype(np.int64)
train_y_aug = train_y[aug_indices]
train_y = np.concatenate((train_y, train_y_aug))


# This architecture is inspired by https://arxiv.org/pdf/1409.6070.pdf
class DeepCNet(nn.Module):
    def __init__(self):
        super(DeepCNet, self).__init__()

        self.device = torch.device("cuda")

        modules = [
            nn.Conv2d(
                in_channels=3, out_channels=400, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=400, out_channels=500, kernel_size=1, stride=1, padding=0
            ),
        ]
        for i in range(4):
            modules.append(
                nn.Conv2d(
                    in_channels=500 + i * 100,
                    out_channels=600 + i * 100,
                    kernel_size=2,
                    stride=1,
                    padding=1,
                )
            )
            modules.append(nn.Dropout(p=0.1 + 0.1 * i))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(kernel_size=2))
            modules.append(
                nn.Conv2d(
                    in_channels=600 + i * 100,
                    out_channels=600 + i * 100,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        modules.append(nn.Dropout(p=0.5))
        modules.append(
            nn.Conv2d(
                in_channels=900, out_channels=2, kernel_size=1, stride=1, padding=0
            )
        )

        self.net = nn.Sequential(*modules).cuda()

    def forward(self, x):
        cx = torch.tensor(x, device=self.device).float()
        out = self.net(cx)

        out = out.reshape(out.size(0), -1)  # turn to 1D array

        return out


# mixup is described here: https://arxiv.org/pdf/1710.09412.pdf
# Adding mixup doesn't change final score, however if preliminary augmentation is
# removed, then the score is around 0.92. Same goes for constant learning rate instead of adaptive.
def mixup_input(x, y, bs):
    l = np.random.beta(1, 1)
    indices = torch.randperm(bs)

    mixed_x = l * x + (1 - l) * x[indices, :]
    y_a, y_b = y, y[indices]

    return mixed_x, y_a, y_b, l


def mixed_criterion(c, l, pred, y_a, y_b):
    return l * c(pred, y_a) + (1 - l) * c(pred, y_b)


# TRAIN PART
modelDeepCNet = DeepCNet().cuda()

batchSize = 10
num_epochs = 75
learning_rate = 1e-4
optimizer = torch.optim.Adam(modelDeepCNet.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("DeepCNet Net")
for epoch in range(num_epochs):
    loss = 0
    acc = 0
    for i in range(0, images_train.shape[0], batchSize):
        input = images_train[i : i + batchSize]
        labels = train_y[i : i + batchSize]
        v = torch.LongTensor(labels).cuda()

        input, y_a, y_b, l = mixup_input(input, v, batchSize)

        output = modelDeepCNet(input)

        # loss = criterion(output, v)
        loss = mixed_criterion(criterion, l, output, y_a, y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(output.data, 1)
        # correct = (predicted.cpu().numpy() == labels).sum().item()
        correct = (
            l * predicted.eq(y_a.data).cpu().sum().float()
            + (1 - l) * predicted.eq(y_b.data).cpu().sum().float()
        )
        acc += correct

    print(
        "Epoch {}, loss {}, accuracy {}, lr {}".format(
            epoch + 1, loss.item(), acc / images_train.shape[0], learning_rate
        )
    )
    learning_rate *= 0.8
    if (epoch + 1) % 15 == 0:
        learning_rate = 1e-4

    optimizer = torch.optim.Adam(modelDeepCNet.parameters(), lr=learning_rate)


# INFERENCE PART

result = {"id": [], "target": []}
classes = ["Airplane", "Bird"]  # categories are sorted in alphabetical order

for i in range(0, images_test.shape[0], batchSize):
    input = images_test[i : i + batchSize]

    output = modelDeepCNet(input)
    predicted = torch.argmax(output.data, 1)

    for idx in range(len(predicted)):
        result["id"].append(i + idx)
        result["target"].append(classes[predicted[idx]])

df = pd.DataFrame(result, columns=result.keys())
df.to_csv(f"{PATH}result_conv.csv", index=False)
