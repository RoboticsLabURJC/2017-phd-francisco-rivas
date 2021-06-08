import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloaders.visual_control_encoder import VisualControlEncoder
from models.visual_control import VisualControl

if __name__ == "__main__":
    dataset_path = "/home/frivas/Descargas/complete_dataset"
    base_size = 418
    batch_size = 40

    encoder_type = VisualControlEncoder.WV_5_3_CLASSES

    model = VisualControl(dataset_path=dataset_path, lr=5e-2,
                          base_size=base_size, batch_size=batch_size, dataset_encoder_type=encoder_type)
    checkpoint_callback_loss = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        filename='rc-classification-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}',
        mode="min")
    if model.encoder.head_type == VisualControlEncoder.CLASSIFICATION_TYPE:
        checkpoint_callback_valid = ModelCheckpoint(
            monitor='val_acc',
            save_top_k=1,
            filename='rc-classification-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}',
            mode="max")
    if model.encoder.head_type == VisualControlEncoder.REGRESSION_TYPE:
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

    trainer = pl.Trainer(gpus=1, max_epochs=2000, progress_bar_refresh_rate=20,
                         callbacks=[checkpoint_callback_loss, checkpoint_callback_valid, checkpoint_callback_train])
    # trainer.tune(model)
    trainer.fit(model)
