import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from net_config.net_config import NetConfig
from models.visual_control import VisualControl

if __name__ == "__main__":
    dataset_path = "/home/frivas/Descargas/complete_dataset"

    network_config_file = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/net_config/MobileSmallRegression.yml"

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
    if model.net_config.head_type == NetConfig.REGRESSION_TYPE:
        checkpoint_callback_valid = ModelCheckpoint(
            monitor='rmse',
            save_top_k=1,
            filename='rc-classification-{epoch:02d}-{rmse:.2f}-{val_loss:.2f}',
            mode="min")
    checkpoint_callback_train = ModelCheckpoint(
        monitor='train_loss',
        save_top_k=1,
        filename='rc-classification-train-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}',
        mode="min")

    trainer = pl.Trainer(gpus=1, max_epochs=100, progress_bar_refresh_rate=20,
                         callbacks=[checkpoint_callback_loss, checkpoint_callback_valid, checkpoint_callback_train])
    # trainer.tune(model)
    trainer.fit(model)
