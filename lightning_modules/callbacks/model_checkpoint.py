"""
SCOB
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""

from pytorch_lightning.callbacks import ModelCheckpoint


class LastestModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        """Save the latest model at every train epoch end."""
        self.save_checkpoint(trainer)
