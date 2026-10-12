'''
Concrete IO class for a specific dataset

CHANGES FROM STAGE 1:
- Reads TWO pre-partitioned files (train + test) instead of one toy file
- Comma-separated values instead of space-separated
- Label is the FIRST column instead of the last
- Returns data in {'train': {...}, 'test': {...}} structure that
  Method_MLP.run() and Setting_Train_Test_Split expect
- No train/test splitting needed (instructor pre-partitioned it)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = '../../data/stage_2_data/'
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    '''
    Reads one CSV file and splits it into features (X) and labels (y).
    '''
    def _load_file(self, file_path):
        X, y = [], []                           # Empty containers for features & labels
        with open(file_path, 'r') as f:         # Open the file and "with" ensures it closes after completion even if error occurs midway
            for line in f:                      #
                line = line.strip()             # removes leading whitespaces and newline characters
                if not line:                    # Skips empty lines
                    continue
                # Split the line into individual string tokens by comma
                # Example: "7,0,0,15,36,..." -> ['7', '0', '0', '15', '36', ...]
                elements = line.split(',')
                try:
                    label = int(elements[0])
                except ValueError:
                    continue
                # Converts all remaining elements to floats using list comprehension
                features = [float(v) for v in elements[1:]]
                X.append(features)
                y.append(label)
        return X, y                 # Return both as tuples

    def load(self):
        print('loading training set...')

        X_train, y_train = self._load_file(
            self.dataset_source_folder_path + self.train_file_name)
        print(f'   train samples: {len(X_train)}, features: {len(X_train[0])}')

        print('--loading testing set...')
        X_test, y_test = self._load_file(
            self.dataset_source_folder_path + self.test_file_name)
        print(f'   test samples: {len(X_test)}, features: {len(X_test[0])}')

        self.data = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }
        return self.data