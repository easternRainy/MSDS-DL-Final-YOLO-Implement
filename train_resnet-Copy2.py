from datasets import *
from learner import *
from pre_resnet import *
from losses import *

lr = 0.001
gpu_id = 2
num_epochs = 100
batch_size = 20
device = torch.device(gpu_id) if torch.cuda.is_available() else "cpu"

df_train, df_valid = load_full_data()

train_ds = YOLODataset(df_train)
valid_ds = YOLODataset(df_valid)

train_dl = DataLoader(train_ds, batch_size, drop_last=False)
valid_dl = DataLoader(valid_ds, batch_size, drop_last=False)

criterion = YOLOLoss()

model = ResnetObj()

optimizer = optim.Adam(model.parameters(), lr=lr)
learner = Learner(model, criterion, optimizer, "resnet_on_full_data")
learner.train(train_dl, valid_dl, device, num_epochs)
