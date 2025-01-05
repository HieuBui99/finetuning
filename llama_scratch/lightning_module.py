import lightning as pl
import torch
from model import LLama2


class LitGPT(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def configure_model(self):
        if self.model is not None:
            return
        
        model = LLama2(self.cfg)
        self.model = torch.compile(model)

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.cfg["weight_decay"]},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg["lr"], betas=(0.9, 0.95), eps=1e-8)
        return optimizer


    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        
        logits = self.model(x)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss