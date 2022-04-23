import os
from typing import Tuple
from torch import Tensor
from vae.model_utils import ModelUtils, train_kmeans
from vae.model import VariationalAutoencoder, INPUT_SHAPE, D
from vae.config import Config
from vae.mode import Mode as M
from vae.dataset import Dataset, DatasetCIFAR10
from vae.preprocess import Preprocessor


def train():
    config = Config()

    config.batch_size[M.TRAIN] = 256
    config.batch_size[M.EVAL] = 512
    # config.batch_size[M.EVAL] = 32
    config.learning_rate = 1e-3
    # config.learning_rate = 1e-5
    config.early_stopping = True
    config.early_stopping_threshold = 20
    config.epochs_per_checkpoint = 10
    config.capacity = 8
    # config.variational_beta = 0
    config.input_shape = INPUT_SHAPE[D.NMINST]
    config.n_clusters = 8
    config.kmeans_model_path = os.path.join(config.log_dir, "nminst", "nminst_8")

    config.display()

    model = VariationalAutoencoder(config)

    train_set = Dataset(M.TRAIN, config)
    valid_set = Dataset(M.EVAL, config)

    # utils = ModelUtils.load_checkpoint(model, path, config)
    # utils = ModelUtils.load_last_checkpoint(model, config)
    utils = ModelUtils.start_new_training(model, config)

    # utils.train(50, train_set, valid_set)
    # utils.plot_history()
    # utils.visualise_output(valid_set)
    # utils.visualise_2d_lantent_space()
    return 

def main():
    train()
    return




if __name__ == "__main__":
    main()
