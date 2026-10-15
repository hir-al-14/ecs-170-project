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

def plot_comparison(loss_dict, save_path, title='Ablation: Training Loss'):
    plt.figure(figsize=(8, 5))
    for label, loss_history in loss_dict.items():
        epochs = range(1, len(loss_history) + 1)
        plt.plot(epochs, loss_history, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    fake_losses = [2.3 / (epoch + 1) + 0.05 for epoch in range(50)]
    plot_learning_curve(fake_losses, 'test_curve.png')
    print('saved test_curve.png')

    fake_dict = {
        'lr = 0.001': fake_losses,
        'lr = 0.01': [loss * 1.5 for loss in fake_losses],
    }
    plot_comparison(fake_dict, 'test_comparison.png')
    print('saved test_comparison.png')