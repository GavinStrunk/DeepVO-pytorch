import time
import torch
from torch.utils.data import DataLoader

from params import par
from model import DeepVO
from kitti_datasets import ImageSequenceDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device: {}'.format(device))

# Create the data loaders
train_dataset = ImageSequenceDataset(dataset_dir=par.dataset_dir,
                                     sequences=par.train_video,
                                     resize=(par.img_w, par.img_h),
                                     num_stack=2)
train_loader = DataLoader(train_dataset,
                          batch_size=par.batch_size,
                          num_workers=par.n_processors,
                          shuffle=True)

test_dataset = ImageSequenceDataset(dataset_dir=par.dataset_dir,
                                    sequences=par.valid_video,
                                    resize=(par.img_w, par.img_h),
                                    num_stack=2)
test_loader = DataLoader(test_dataset,
                         batch_size=par.batch_size,
                         num_workers=par.n_processors,
                         shuffle=True)

# Load the model
model = DeepVO(par.img_h, par.img_w, par.batch_norm).to(device)
model.train()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

min_test_loss = 1e10
for ep in range(par.epochs):
    start = time.time()

    model.train()

    loss_mean = 0
    loss_list = []
    for idx, (data, label) in enumerate(train_loader):
        loss = model.step(data.to(device), label.to(device), optimizer)

        loss_list.append(loss)
        loss_mean += float(loss)

    loss_mean /= len(train_loader)
    print('Train Epoch {}/{} Elapse Time: {} Mean Loss: {}'.format(ep, par.epochs, time.time() - start, loss_mean))

    test_start = time.time()
    model.eval()
    test_loss_mean = 0
    test_loss_list = []
    for idx, (data, label) in enumerate(test_loader):
        loss = model.get_loss(data.to(device), label.to(device)).data.cpu().numpy()

        test_loss_list.append(loss)
        test_loss_mean += float(loss)

    test_loss_mean /= len(test_loader)
    print('Test Elapse Time: {} Mean Loss: {}'.format(time.time() - start, loss_mean))

    # Save model if the test loss decreases
    if test_loss_mean < min_test_loss:
        torch.save(model.state_dict(), 'models/deepvo-ep{}.pt'.format(ep))
