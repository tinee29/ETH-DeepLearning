import torch
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from .metrics import accuracy_loss
from .utils import resolve_optimizer, resolve_lr_scheduler

class Experiment(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        #self.save_hyperparameters()
        
        self.cfg = cfg

        self.model = cfg["model"]
        if cfg["dataset"] != "cifar10":
            num_feat = self.model.fc.in_features
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_feat, num_feat),
                torch.nn.ReLU(),
                torch.nn.Linear(num_feat, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, cfg["num_classes"])                                                
            )
        
        self.loss = cfg["loss"]

    def forward(self, inputs):
        out = self.model(inputs)
        return out
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)        
        loss = self.loss(out.squeeze(), y.squeeze())        
        self.log("train_loss", loss)
        acc = accuracy_score(y.cpu().numpy(), out.cpu().detach().numpy().argmax(axis=-1))
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        if self.cfg["setup"] == "retain":
            acc, loss = accuracy_loss(self.model, self.cfg["forget_loader"])
            self.log("forget_accuracy", acc)
            self.log("forget_loss", loss)
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.loss(out.squeeze(), y.squeeze())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = accuracy_score(y.cpu().numpy(), out.cpu().numpy().argmax(axis=-1))
        self.log("val_accuracy", acc, on_step=False, on_epoch=True)
        
        
    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.cfg, self.parameters()) if "only_train_fc" not in self.cfg.keys() or not self.cfg["only_train_fc"] else resolve_optimizer(self.cfg, self.model.fc.parameters())
        lr_scheduler = resolve_lr_scheduler(self.cfg, optimizer)
        if lr_scheduler is None:
            return [optimizer]
        return [optimizer], [lr_scheduler]
