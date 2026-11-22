import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# hiral saved the losses here after training
here = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(here, '..', '..', 'result', 'stage_4_result', 'generation', 'rnn_review_baseline')

with open(os.path.join(result_dir, 'history.json')) as f:
    history = json.load(f)

train_loss = history['train_loss']
val_loss = history['validation_loss']
epochs = range(1, len(train_loss) + 1)

# train and validation on the same plot so we can see overfitting
plt.figure()
plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('RNN Text Generation Learning Curve')
plt.legend()
plt.grid(True)

save_path = os.path.join(here, '..', '..', 'result', 'stage_4_result', 'generation_learning_curve.png')
plt.savefig(save_path)
print('saved', save_path)