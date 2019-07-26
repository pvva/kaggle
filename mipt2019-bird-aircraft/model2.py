import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
import torch
import torch.nn as nn


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


# maybe it makes sense to use conv with dilation instead of pooling
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        ).cuda()
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU()
        ).cuda()
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.LPPool2d(2, kernel_size=3, stride=1),
        ).cuda()
        self.drop_out = nn.Dropout(p=0.5).cuda()
        self.layer4 = nn.Sequential(
            nn.Linear(36864, 512),
            nn.ReLU(),
            #
        ).cuda()
        self.layer5 = nn.Sequential(
            nn.Linear(512, 2),
            #
        ).cuda()
        self.device = torch.device("cuda")

    def forward(self, x):
        cx = torch.tensor(x, device=self.device).float()
        out = self.layer1(cx)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)  # turn to 1D array
        out = self.layer4(out)
        out = self.drop_out(out)
        out = self.layer5(out)

        return out


# TRAIN PART

batchSize = 20
model = ConvNet().cuda()

num_epochs = 50
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
