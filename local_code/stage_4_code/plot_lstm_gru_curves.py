import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# chaemin saved the training logs for lstm and gru here
here = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(here, '..', '..', 'result', 'stage_4_result')

for name in ['lstm', 'gru']:
    with open(os.path.join(result_dir, name + '_classification_training_log.json')) as f:
        log = json.load(f)

    epochs = range(1, len(log['train_loss']) + 1)

    # train and test loss together so we can see when it starts overfitting
    plt.figure()
    plt.plot(epochs, log['train_loss'], label='train loss')
    plt.plot(epochs, log['test_loss'], label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(name.upper() + ' Text Classification Learning Curve')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(result_dir, name + '_classification_curve.png')
    plt.savefig(save_path)
    plt.close()
    print('saved', save_path)