import torch


def make_dataset():
    """Return train and test dataloaders for MNIST."""

    train_data, train_labels = (
        [],
        [],
    )

    for i in range(5):
        train_data.append(torch.load(f"data/raw/corruptmnist/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/corruptmnist/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("data/raw/corruptmnist/test_images.pt")
    test_labels = torch.load("data/raw/corruptmnist/test_target.pt")

    print(train_data.shape)
    print(train_labels.shape)

    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    train_data = (train_data - train_data.mean()) / train_data.std()
    test_data = (test_data - test_data.mean()) / test_data.std()

    train = torch.utils.data.TensorDataset(train_data, train_labels)
    test = torch.utils.data.TensorDataset(test_data, test_labels)

    torch.save(train, "data/processed/train_dataset.pt")
    torch.save(test, "data/processed/test_dataset.pt")


if __name__ == "__main__":
    make_dataset()
