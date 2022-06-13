import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from net_config.net_config import NetConfig
from models.visual_control import VisualControl

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network given a configuration file.')
    parser.add_argument('--config_file', type=str, help='Input configuration file')

    return parser.parse_args()

if __name__ == "__main__":

    arguments = parse_args()
    dataset_path = "/home/frivas/Descargas/complete_dataset"
    dataset_path = "/home/frivas/Descargas/datasets_opencv"
    # dataset_path = "/home/frivas/devel/shared/mio/datasets_opencv"
    # dataset_path = "/home/frivas/devel/mio/phd/dataset/datasets_opencv"
    # dataset_path = "/home/frivas/devel/shared/mio/datasets_opencv"
    # dataset_path = "/home/frivas/devel/mio/phd/dataset/datasets_opencv"


    network_config_file = arguments.config_file
    net_config = NetConfig(network_config_file)

    model = VisualControl(dataset_path=dataset_path, lr=5e-2,
                          base_size=net_config.base_size, batch_size=net_config.batch_size, net_config=net_config)
    checkpoint_callback_loss = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        filename='rc-classification-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}',
        mode="min")
    if model.net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
        checkpoint_callback_valid = ModelCheckpoint(
            monitor='val_acc',
            save_top_k=1,
            filename='rc-classification-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}',
            mode="max")
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=50,
                                            verbose=True, mode="max")
    if model.net_config.head_type == NetConfig.REGRESSION_TYPE:
        checkpoint_callback_valid = ModelCheckpoint(
            monitor='rmse',
            save_top_k=1,
            filename='rc-classification-{epoch:02d}-{rmse:.2f}-{val_loss:.2f}',
            mode="min")
        early_stop_callback = EarlyStopping(monitor="rmse", min_delta=0.00, patience=50,
                                            verbose=True, mode="min")
    checkpoint_callback_train = ModelCheckpoint(
        monitor='train_loss',
        save_top_k=1,
        filename='rc-classification-train-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}',
        mode="min")


    trainer = pl.Trainer(gpus=1, max_epochs=300, progress_bar_refresh_rate=20,
                         callbacks=[checkpoint_callback_loss, checkpoint_callback_valid, checkpoint_callback_train, early_stop_callback])


    # trainer.tune(model)
    trainer.fit(model)
