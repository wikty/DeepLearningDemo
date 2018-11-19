import config
from load_dataset import Loader
from lib.utils import Params

data_dir = config.data_dir
datasets_params_file = config.datasets_params_file
datasets_params = Params(datasets_params_file)
loader = Loader(data_dir, datasets_params, 'utf8')

i = 0
dataset = loader.load('train', datasets_params.train_size, batch_size=1)
for batch in dataset:
    i += 1

print(i)