import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# Competition: https://www.kaggle.com/c/mipt2019-bird-aircraft
# This model gives score of 0.9


def convert_strings_to_categories_codes(df):
    for n, c in df.items():
        if is_string_dtype(c):
            df[n] = c.astype("category").cat.codes


PATH = "C:/Dev/data/mipt2019-bird-aircraft/"
train_x = pd.read_csv(f"{PATH}train_x.csv", index_col=0, header=None)
train_y = pd.read_csv(f"{PATH}train_y.csv", index_col=0)
test_x = pd.read_csv(f"{PATH}test_x.csv", index_col=0, header=None)

images_train = (
    train_x.values.reshape(train_x.shape[0], 32, 32, 3).swapaxes(1, 3).swapaxes(2, 3)
)

images_test = (
    test_x.values.reshape(test_x.shape[0], 32, 32, 3).swapaxes(1, 3).swapaxes(2, 3)
)

convert_strings_to_categories_codes(train_y)
train_y = train_y["target"].to_numpy().astype(np.int64)


# Different types of intermediate pooling techniques affects score:
# max pooling (kernel = 2): 0,88229
# conv2d (stride = 2): 0,86
# avgpool (kernel = 2): 0,875
# lppool (squared, kernel = 2): 0,872
# no pooling in between, maxppol (kernel = 12) at the end: 0,898
# no pooling, zoom out convs at the end: 0,898
# no pooling, zoom out convs at the end, weight decay = 1e-5: 0,901
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.drop_out = nn.Dropout(p=0.5).cuda()
        self.final_layer = nn.Sequential(
            nn.Linear(4096, 2),
            #
        ).cuda()
        self.device = torch.device("cuda")
        self.extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=2, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=0
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(
            #     in_channels=16, out_channels=16, kernel_size=2, stride=2, padding=0
            # ),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(
            #     in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0
            # ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(
            #     in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0
            # ),
            # nn.ReLU(),
        ).cuda()
        self.zoomout = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=8, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=12, stride=1, padding=0
            ),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=12),
        )

    def forward(self, x):
        cx = torch.tensor(x, device=self.device).float()
        out = self.extractor(cx)
        out = self.zoomout(out)

        out = out.reshape(out.size(0), -1)  # turn to 1D array
        out = self.drop_out(out)  # 0.885 with dropout
        out = self.final_layer(out)

        return out


# TRAIN PART

batchSize = 50
model = ConvNet().cuda()

num_epochs = 60
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    loss = 0
    acc = 0
    for i in range(0, images_train.shape[0], batchSize):
        input = images_train[i : i + batchSize]
        labels = train_y[i : i + batchSize]

        output = model(input)
        v = torch.LongTensor(labels).cuda()

        loss = criterion(output, v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        correct = (predicted.cpu().numpy() == labels).sum().item()
        acc += correct

    print(
        "Epoch {}, loss {}, accuracy {}, lr {}".format(
            epoch + 1, loss.item(), acc / images_train.shape[0], learning_rate
        )
    )

# INFERENCE PART

result = {"id": [], "target": []}
classes = ["Airplane", "Bird"]  # categories are sorted in alphabetical order

for i in range(0, images_test.shape[0], batchSize):
    input = images_test[i : i + batchSize]

    output = model(input)
    _, predicted = torch.max(output.data, 1)

    for idx in range(len(predicted)):
        result["id"].append(i + idx)
        result["target"].append(classes[predicted[idx]])

df = pd.DataFrame(result, columns=result.keys())
df.to_csv(f"{PATH}result_conv.csv", index=False)

# VISUALIZE KERNELS

# kernels = model.cpu().extractor[0].weight.detach().clone()
# kernels = kernels - kernels.min()
# kernels = kernels / kernels.max()
# img = make_grid(kernels)
# plt.imshow(img.permute(1, 2, 0))
# plt.show()

# kernels = model.cpu().extractor[3].weight.detach().clone()
# kernels = kernels - kernels.min()
# kernels = kernels / kernels.max()
# k0 = kernels[0:64, 0:3, :, :]
# img = make_grid(k0)
# plt.imshow(img.permute(1, 2, 0))
# plt.show()
