import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class SparseCAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = [128, 64],
        lambda_sparse: float = 1e-3,
        lambda_cae: float = 1e-4,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Encoder ──────────────────────────────────────────────
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim), nn.Sigmoid()]
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──────────────────────────────────────────────
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    # ─────────────────────────────────────────────────────────────
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    # ── Regularizaciones ─────────────────────────────────────────
    def _sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return z.abs().mean()

    @torch.enable_grad()
    def _cae_loss(self, z, x):
        x = x.detach().requires_grad_(True)
        z = self.encoder(x)
        frob_sq = torch.zeros(1, device=x.device)
        for i in range(z.size(1)):
            (grad,) = torch.autograd.grad(
                z[:, i].sum(), x, create_graph=True, retain_graph=True
            )
            frob_sq += (grad ** 2).sum()
        return frob_sq / x.size(0)

    # ── Paso compartido ──────────────────────────────────────────
    def _shared_step(self, batch, stage: str):
        x, _ = batch
        z     = self.encoder(x)
        x_hat = self.decoder(z)

        recon  = F.mse_loss(x_hat, x)
        sparse = self._sparsity_loss(z)
        loss   = recon + self.hparams.lambda_sparse * sparse

        if stage == "train":
            cae  = self._cae_loss(z, x)
            loss = loss + self.hparams.lambda_cae * cae
            self.log("train/cae", cae, on_epoch=True, on_step=False)

        self.log_dict(
            {f"{stage}/loss": loss, f"{stage}/recon": recon, f"{stage}/sparse": sparse},
            prog_bar=True, on_epoch=True, on_step=False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    # ── Optimizador ──────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }