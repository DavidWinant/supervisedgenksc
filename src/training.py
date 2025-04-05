import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from supervised_ksc_model import RKM_Stiefel, stiefel_opti
from prepare_data import get_dataloader, get_gaussian_dataset
from types import SimpleNamespace
import os
import datetime


class RKMStiefelLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.automatic_optimization = False

        # Get input dimensions from dataloader
        data_loader = get_dataloader(config)
        dataset = data_loader.dataset
        input_dim = dataset[0][0].size(0)

        # n_channels = dataset[0][0].size[0]

        # Initialize the model
        self.model = RKM_Stiefel(ipVec_dim=input_dim, args=config)

        # For updating cluster codes
        self.latest_batch = None

    def forward(self, x):
        # Forward pass through the model
        if len(x.shape) == 2 and False:
            x = x.view(-1, 1, 28, 28)  # Reshape if needed


        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Get optimizers
        stiefel_optimizer, net_optimizer = self.optimizers()
        points, labels = batch
        if batch_idx == 0:
            self.latest_batch = points
        batch_size = points.size(0)

        # Forward pass - expect the model to return intermediate values, not losses
        phi, e, h, x_tilde, dcos, Dinv = self.model.forward(points)

        # Calculate individual loss components
        ksc_loss = self._calculate_ksc_loss(phi, Dinv, batch_size)
        recon_loss = self._calculate_reconstruction_loss(x_tilde, batch, batch_size)
        cosine_loss = self._calculate_cosine_distance_loss(dcos, labels, batch_size)
        unbalance_loss = self._calculate_unbalance_loss(e, batch_size)

        # Apply weights during summation
        ksc_weight = self.config.c_ksc * 10
        recon_weight = self.config.c_accu
        cosine_weight = self.config.c_clust * (1 - self.config.c_balance)
        unbalance_weight = self.config.c_clust * self.config.c_balance

        print("Type")
        print(type(cosine_loss))
        # Total loss with explicit weighting
        loss = (
            ksc_weight * ksc_loss
            + recon_weight * recon_loss
            + cosine_weight * cosine_loss
            + unbalance_weight * unbalance_loss
        )

        # Optimization
        stiefel_optimizer.zero_grad()
        net_optimizer.zero_grad()
        self.manual_backward(loss)
        stiefel_optimizer.step()
        net_optimizer.step()

        # Logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("ksc_loss", ksc_loss, on_step=False, on_epoch=True)
        self.log("recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("cosine_loss", cosine_loss, on_step=False, on_epoch=True)
        self.log("unbalance_loss", unbalance_loss, on_step=False, on_epoch=True)

        return loss

    def _calculate_ksc_loss(self, phi, Dinv, N):
        return (
            (
                1 / 2 * self.config.h_dim
                - 1
                / 2
                * torch.trace(
                    self.model.Ut @ (phi.t() * Dinv) @ phi @ self.model.Ut.t()
                )
                + 1 / 2 * torch.trace(phi.t() @ phi)
            )
        ) / N

    def _calculate_reconstruction_loss(self, x_tilde, batch, N):
        points, _ = batch
        return (
            0.5
            * (
                self.model.recon_loss(
                    x_tilde.view(-1, self.model.ipVec_dim),
                    points.view(-1, self.model.ipVec_dim),
                )
            )
            / N
        )

    def _calculate_cosine_distance_loss(
        self, dcos: torch.Tensor, labels: torch.Tensor, N: int
    ) -> tuple[torch.Tensor,torch.Tensor]:
        # Create a mask where True indicates positions that should be set to infinity
        # First create a matrix where each row has the label's index set to False, all others True

        final_dcos = torch.zeros_like(labels, dtype=dcos.dtype)
        unlabelled_mask = torch.isnan(labels)
        labelled_mask = ~unlabelled_mask
        
        if labelled_mask.any():
            final_dcos[labelled_mask] = dcos[labelled_mask, labels[labelled_mask].long()]
        
        if unlabelled_mask.any():
            final_dcos[unlabelled_mask] = torch.min(dcos[unlabelled_mask], dim=1).values

        oneN = torch.ones(N, 1).to(self.config.proc)
        cosine_distance_loss = oneN.t() @ final_dcos
        return cosine_distance_loss / N

    def _calculate_unbalance_loss(self, e, N):
        z = e[:, : self.config.k - 1]
        unbalance_loss = (z.t() / z[:, : self.config.k - 1].norm(dim=1)).sum() / N
        return unbalance_loss.pow(2)

    def on_train_epoch_end(self):
        # Update cluster codes periodically
        if self.current_epoch % 10 == 0 and self.latest_batch is not None:
            batch_data = self.latest_batch

            if len(batch_data.shape) == 2 and False:
                batch_data = batch_data.view(-1, 1, 28, 28)

            # with torch.no_grad():
            #     phi = self.model.encoder(batch_data)
            #     self.model.update_cluster_codes(phi)

    def configure_optimizers(self):
        # Setup optimizers as in the original code
        stief_param = [self.model.Ut]
        net_params = [p for name, p in self.model.named_parameters() if name != "Ut"]

        stiefel_optimizer = stiefel_opti(stief_param, lrg=self.config.lr)
        net_optimizer = torch.optim.Adam(net_params, lr=self.config.lr)

        # Return optimizers in a format PyTorch Lightning recognizes
        return [stiefel_optimizer, net_optimizer]


class RKMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_loader = get_dataloader(self.config, "gaussian")

    def train_dataloader(self):
        return self.train_loader


def load_config(config_path) -> SimpleNamespace:
    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
            import yaml  # Using PyYAML

            config = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config file format. Use json or yaml.")
    return SimpleNamespace(**config)


def train_model(config_path="config/config.yaml"):
    # Load configuration
    config = load_config(config_path)

    # Set seed for reproducibility
    pl.seed_everything(config.seed if hasattr(config, "seed") else 42)

    # Create data module
    dm = RKMDataModule(config)

    # Create model
    model = RKMStiefelLightning(config)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{train_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="train_loss",
        mode="min",
    )

    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="rkm_stiefel")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if config.proc == "cuda" else "cpu",
        devices=1,  # Set to 1 for both GPU and CPU
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    # Train the model
    print("Starting training...")
    trainer.fit(model, datamodule=dm)

    # Create timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model with timestamp
    model_filename = f"rkm_stiefel_model_{timestamp}.pth"
    torch.save(model.model, model_filename)
    
    # Save model attributes with timestamp
    attributes_filename = f"rkm_stiefel_attributes_{timestamp}.pth"
    model_config = {
        'config': vars(model.config),
        'timestamp': timestamp
    }
    torch.save(model_config, attributes_filename)
    
    print(f"Model saved as {model_filename}")
    print("Training completed!")

    return model

def load_model(model_path="rkm_stiefel_model.pth", attributes_path="rkm_stiefel_attributes.pth"):
    """
    Load a saved RKM Stiefel model and its attributes.
    
    Args:
        model_path (str): Path to the saved model state dict
        attributes_path (str): Path to the saved model attributes
        
    Returns:
        tuple: (loaded_model, model_attributes)
    """
    # Load model attributes first to get configuration
    model_attributes = torch.load(attributes_path, weights_only=False)
    config = SimpleNamespace(**model_attributes['config'])
    
    # Get input dimensions - we need to reconstruct this
    # If loading for inference, you may need to specify input dimension manually

    # Load state dictionary
    model = torch.load(model_path)
    
    # Put model in evaluation mode    
    print(f"Model loaded successfully from {model_path}")
    print(f"Model attributes loaded from {attributes_path}")
    
    return model, model_attributes

if __name__ == "__main__":
    train_model()
