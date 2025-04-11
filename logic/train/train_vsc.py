import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
from logic.common import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_VSC_DELTA_LAMBDA, DEFAULT_VSC_L, DEFAULT_VSC_N_WARMUP, PATH_VSC, PATH_VSC_WARMUP
from logic.model.vsc import VSC
from logic.train.base import process


def train_vsc_standard(
        model: nn.Module,
        dataloader: DataLoader,
        epochs: int,
        device: torch.device,
        verbose: bool = True
) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_sparsity_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, gamma = model(x)
            # Reconstruction loss: binary cross-entropy, somme sur pixels, moyenne sur batch.
            recon_loss = nn.functional.binary_cross_entropy(
                x_recon, x, reduction='none').sum(dim=[1, 2, 3]).mean()
            # KL divergence pour la partie continue (slab).
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) -
                              torch.exp(logvar)).sum(dim=1).mean()
            # Pénalité de sparsité (doit être calculée via divergence KL entre Bernoulli(gamma) et Bernoulli(alpha)).
            sparsity_loss = model.compute_sparsity_loss(gamma)
            loss = recon_loss + kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_sparsity_loss += sparsity_loss.item()
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_sparsity_loss = total_sparsity_loss / len(dataloader)
        history.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_recon_loss': avg_recon_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_sparsity_loss': avg_sparsity_loss
        })
        if verbose:
            print(
                f"Standard Training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return history


def train_vsc_warmup(model: VSC,
                     dataloader: DataLoader,
                     epochs: int,
                     device: torch.device,
                     verbose: bool = True,
                     n_warmup: int = DEFAULT_VSC_N_WARMUP,
                     delta_lambda: float = DEFAULT_VSC_DELTA_LAMBDA,
                     L: int = DEFAULT_VSC_L
                     ) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    model.train()
    history = []
    # Initialize lambda for warm-up.
    model.lambda_val = 0.0
    iteration = 0
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_sparsity_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            # Reconstruction L on sample.
            recon_losses = []
            for _ in range(L):
                x_recon, mu, logvar, gamma = model(x)
                recon_loss = nn.functional.binary_cross_entropy(
                    x_recon, x, reduction='none').sum(dim=[1, 2, 3])
                recon_losses.append(recon_loss)
            recon_loss = torch.stack(recon_losses, dim=0).mean()
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) -
                              torch.exp(logvar)).sum(dim=1).mean()
            sparsity_loss = model.compute_sparsity_loss(gamma)
            loss = recon_loss + kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_sparsity_loss += sparsity_loss.item()

        # Warm-up : Increase warmup up to n_warmup
        if iteration < n_warmup:
            model.lambda_val = min(1.0, model.lambda_val + delta_lambda)
            iteration += 1
        else:
            model.lambda_val = 1.0

        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_sparsity_loss = total_sparsity_loss / len(dataloader)
        history.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_recon_loss': avg_recon_loss,
            'avg_kl_loss': avg_kl_loss,
            'avg_sparsity_loss': avg_sparsity_loss,
            'lambda': model.lambda_val
        })

        if verbose:
            print(
                f"Warmup Training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, lambda: {model.lambda_val:.2f}")
    return history


def process_vsc(
        model: VSC,
        dataloader: DataLoader,
        device: torch.device,
        no_cache: bool = False,
        model_path: str = PATH_VSC,
        epochs: int = DEFAULT_EPOCHS) -> Tuple[List[dict], VSC]:
    return process(model_path, train_vsc_standard, model, dataloader, device, no_cache, epochs)


def process_vsc_warmup(
        model: VSC,
        dataloader: DataLoader,
        device: torch.device,
        no_cache: bool = False,
        model_path: str = PATH_VSC_WARMUP,
        epochs: int = DEFAULT_EPOCHS,
        n_warmup: int = DEFAULT_VSC_N_WARMUP,
        delta_lambda: float = DEFAULT_VSC_DELTA_LAMBDA,
        L: int = DEFAULT_VSC_L
) -> Tuple[List[dict], VSC]:
    def training_function(m, d, e, dev): return train_vsc_warmup(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        device=device,
        verbose=True,
        n_warmup=n_warmup,
        delta_lambda=delta_lambda,
        L=L)
    return process(model_path, training_function, model, dataloader, device, no_cache, epochs)
