import torch
import click
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mnist_classifier import MyNeuralNet


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@click.command()
@click.option('-model_path', prompt='Model path', default='models/test/model.pth', help='Path to the model.')
def visualize(model_path='models/test/model.pt'):
    """
    Visualizes the feature space of a neural network model using t-SNE.

    This function loads a trained neural network model, processes a batch of data through
    a truncated version of this model to obtain feature representations, and then visualizes
    these features using t-SNE.

    Args:
        model_path (str): The file path where the trained model is stored.
    
    The function creates a 2D t-SNE plot of the features and saves it in a specified directory.
    """
    # Extracting the folder name from the model path
    model_folder = model_path.split("/")[1]

    # Load the neural network model
    net = MyNeuralNet().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # Load the training dataset and get one batch
    train_set = torch.load("data/processed/train_dataset.pt")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    image, label = next(iter(train_loader))

    # Process images through the network
    out = net.leaky_relu(net.conv1(image.to(device)))
    out = net.leaky_relu(net.conv2(out))
    out = net.max_pool2d(out)
    pred = net(image.to(device)).cpu().detach()



    # Perform t-SNE to reduce dimensionality to 2D for visualization
    tsne = TSNE(n_components=2, n_iter=300)
    tsne_results = tsne.fit_transform(pred.detach().numpy())

    # Plot using seaborn
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=label,
        palette=sns.color_palette("hls", 10),
        alpha=0.6
    )
    plt.title(f"t-SNE for model from {model_folder}")
    plt.legend()

    # Save the plot
    plt.savefig(f"reports/figures/{model_folder}/tsne.pdf")

if __name__ == "__main__":
    visualize()