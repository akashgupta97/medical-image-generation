import json
import cv2
import numpy as np

from torch.utils.data import Dataset

import sys
sys.path.append("ControlNet/")
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from ControlNet.cldm.logger import ImageLogger
from ControlNet.cldm.model import create_model, load_state_dict

class MyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            list_data = f.readlines()
            self.data = [json.loads(a) for a in list_data]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        dim = (512,512)
        print()
        source = cv2.resize(source, dim, interpolation = cv2.INTER_CUBIC)
        target = cv2.resize(target, dim, interpolation = cv2.INTER_CUBIC)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    

class EveryNStepsModelCheckpoint(ModelCheckpoint):
    def __init__(self, save_every_n_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.save_every_n_steps = save_every_n_steps
        self.current_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.current_step += 1
        if self.current_step % self.save_every_n_steps == 0:
            filepath = self._get_metric_interpolated_filepath_name(trainer, pl_module)
            self._save_model(trainer, pl_module, filepath)    
    
    
# Configs
resume_path = '/home/jupyter/gcs/checkpoints1/control_sd21_ini.ckpt'
batch_size = 10
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('ControlNet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

checkpoint_callback = ModelCheckpoint(
    monitor="global_step",
    dirpath='checkpoints',
    filename='model-{epoch:02d}-{global_step:.2f}',
    mode='max',
    every_n_train_steps=1000  # checkpoint every N training steps
)



# Misc
train_dataset = MyDataset("/home/jupyter/gcs/train.txt")
val_dataset = MyDataset("/home/jupyter/gcs/val.txt")
train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback])


# Train!
trainer.fit(model, train_dataloader, val_dataloader)