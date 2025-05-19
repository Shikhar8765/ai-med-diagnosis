from dataset import MedDataset
from torch.utils.data import DataLoader

ds = MedDataset("../1_data/train.csv", "../1_data/images")
dl = DataLoader(ds, batch_size=4, shuffle=True)

x, y = next(iter(dl))
print("Batch shape:", x.shape)
print("Labels:", y)
