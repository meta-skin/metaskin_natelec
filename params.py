import attr


@attr.s(auto_attribs=True)
class TDCParams:

    data_dir: str = "data/data"
    train_data_dir: str = "data/data/train_data"

    # make sure to change this to your own path
    pretrain_model_path: str = ""

    # model hyperparameters
    d_model: int = 32
    d_embedding: int = 16
    n_head: int = 4
    n_layers: int = 3
    window_size: int = 32

    # training hyperparameters
    lr: float = 0.0001
    tol_dist: int = 4
    batch_size: int = 100
    num_workers: int = 4
    num_epoch: int = 5000

    # data augmentation hyperparameters
    jittering_mean: float = 0.0
    jittering_std: float = 0.1

    # transfer hyperparameters
    transfer_batch_size: int = 100
    transfer_num_workers: int = 4
    transfer_num_epoch: int = 20

    transfer_loss_alpha: float = 0.5
    transfer_loss_beta: float = 0.01