import torch
import argparse
import json
from supervised_ksc_model import RKM_Stiefel, stiefel_opti
from dataloader import get_dataloader
from tqdm import tqdm
from types import SimpleNamespace



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


def train(model, train_loader, stiefel_optimizer, net_optimizer, args):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(args.proc)
        if len(data.shape) == 2:
            data = data.view(-1, 1, 28, 28)  # Reshape if needed

        # Forward pass
        f1, f2, f3, f4, phi, e, h = model(data)
        loss = f1 + f2 + f3 + f4

        # Backward pass and optimization
        stiefel_optimizer.zero_grad()
        net_optimizer.zero_grad()
        loss.backward()
        stiefel_optimizer.step()
        net_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def main():

    config = load_config('config/config.yaml')

    # Get data
    train_loader, input_dim, n_channels = get_dataloader(config)

    # Initialize model
    model = RKM_Stiefel(ipVec_dim=input_dim, args=config, nChannels=n_channels).to(
        config.proc
    )

    # Setup optimizers
    stief_param = [model.Ut]
    net_params = [p for name, p in model.named_parameters() if name != "Ut"]

    stiefel_optimizer = stiefel_opti(stief_param, lrg=config.lr)
    net_optimizer = torch.optim.Adam(net_params, lr=config.lr)

    # Training loop
    print("Starting training...")
    for epoch in range(config.epochs):
        avg_loss = train(model, train_loader, stiefel_optimizer, net_optimizer, config)
        print(f'Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}')

        # Optional: Update cluster codes periodically
        if epoch % 10 == 0:
            with torch.no_grad():
                batch_data = next(iter(train_loader))[0].to(config.proc)
                if len(batch_data.shape) == 2:
                    batch_data = batch_data.view(-1, 1, 28, 28)
                phi = model.encoder(batch_data)
                model.update_cluster_codes(phi)

    print("Training completed!")

    # Save the model
    torch.save(model.state_dict(), "rkm_stiefel_model.pth")


if __name__ == "__main__":
    main()
