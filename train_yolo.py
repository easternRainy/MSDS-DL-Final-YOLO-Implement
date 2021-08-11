from datasets import *
from learner import *
from yolo import *
from losses import *

lr = 0.001
gpu_id = 0
num_epochs = 2
device = torch.device(gpu_id) if torch.cuda.is_available() else "cpu"

df_train, df_valid = load_tiny()

train_ds = YOLODataset(df_train)
valid_ds = YOLODataset(df_valid)

train_dl = DataLoader(train_ds, batch_size=3, drop_last=False)
valid_dl = DataLoader(valid_ds, batch_size=3, drop_last=False)

criterion = YOLOLoss()

model = Yolo(split_size=7, num_boxes=2, num_classes=20)

optimizer = optim.Adam(model.parameters(), lr=lr)
learner = Learner(model, criterion, optimizer, "yolo_on_tiny")
learner.train(train_dl, valid_dl, device, num_epochs)
