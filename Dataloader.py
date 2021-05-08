import numpy as np

class Dataloader():
    def __init__(self, dataset, batch_size=20, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        data_ids = []
        data_masks = []
        data_labels = []

        for k in range(self.batch_size):
            data_ids.append(data[k][0])
            data_masks.append(data[k][1])
            data_labels.append(data[k][2])

        return np.array(data_ids), np.array(data_masks), np.array(data_labels)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)