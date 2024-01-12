import datetime
import os

import click
import matplotlib.pyplot as plt
import torch

from mnist_classifier import MyNeuralNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@click.command()
@click.option("--lr", default=1e-4, help="learning rate to use for training")
@click.option("--batch_size", default=64, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    # TODO: Implement training loop here
    net = MyNeuralNet().to(device)
    train_set = torch.load("data/processed/train_dataset.pt")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_list = []

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss {loss}")
        loss_list.append(loss.cpu().detach().numpy())  # Store loss value

    # Save the trained model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_savepath = "models/" + timestamp
    os.makedirs(model_savepath)
    torch.save(net.state_dict(), f"{model_savepath}/model.pth")

    # Plot and save the loss graph
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss over Epochs (Model from {timestamp})")

    fig_savepath = "reports/figures/" + timestamp
    os.makedirs(fig_savepath)
    plt.savefig(f"{fig_savepath}/loss.pdf")


if __name__ == "__main__":
    train()
