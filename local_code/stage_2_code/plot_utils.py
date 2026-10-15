import matplotlib.pyplot as plt


def plot_learning_curve(loss_history, save_path, title='Training Loss'):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    fake_losses = [2.3 / (epoch + 1) + 0.05 for epoch in range(50)]
    plot_learning_curve(fake_losses, 'test_curve.png')
    print('saved test_curve.png')