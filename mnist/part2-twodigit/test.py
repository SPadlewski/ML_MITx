import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True
X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

print(y_train[0].shape)