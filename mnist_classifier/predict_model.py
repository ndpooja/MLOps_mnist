import click
import torch

from mnist_classifier.models import model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@click.command()
@click.option("-model_path", prompt="Model path", help="Path to the model.")
@click.option("-data_path", prompt="Data path", help="Path to the data to be processed.")
def predict(model_path, data_path):
    """Predict on a given model and data."""
    print("Predicting model on data")

    # Load the neural network model
    net = model.MyNeuralNet().to(device)
    print(model_path)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    test_set = torch.load(data_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            predicted = outputs.argmax(dim=1).cpu()
            test_preds.append(predicted)
            test_labels.append(labels.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    accuracy = (test_preds == test_labels).float().mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    predict()
