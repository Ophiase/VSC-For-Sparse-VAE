import argparse
import os
import torch
from logic.common import PATH_GRAPHICS, PATH_AE, PATH_VAE, PATH_VSC, PATH_VSC_WARMUP
from logic.data import get_train_dataloader, get_test_dataloader
from logic.model.autoencoder import Autoencoder
from logic.model.base import load_model
from logic.model.vae import VAE
from logic.model.vsc import VSC
from graphics.latent_space import visualize_latent_space
from graphics.reconstruction import visualize_reconstruction
from graphics.activated_dimensions import visualize_activated_dimensions

VERBOSE = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate graphics for trained models.")
    parser.add_argument(
        "--models",
        type=str,
        default="autoencoder,vae,vsc,vsc_warmup",
        help="Comma-separated list of models to visualize (e.g., autoencoder,vae,vsc,vsc_warmup)"
    )
    parser.add_argument(
        "--to_browser",
        action="store_true",
        help="Show figures in browser instead of saving to files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings if models are not found"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to use: mnist or fashionmnist"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Data split: train or test"
    )
    return parser.parse_args()


def generate_graphics(model, name, dataloader, device, to_browser):
    if VERBOSE:
        print(f"- Process <{name}> ...")
    os.makedirs(os.path.join(PATH_GRAPHICS, name), exist_ok=True)
    latent_dim = getattr(model, 'latent_dim', 2)
    graphics = [
        (
            visualize_latent_space,
            {"model": model, "latent_dim": latent_dim,
             "device": device, "text": f"{name} Latent Space"},
            "latent_space.png"
        ),
        (
            visualize_reconstruction,
            {"model": model, "dataloader": dataloader,
             "device": device, "text": f"{name} Reconstruction"},
            "reconstruction.png"
        ),
        (
            visualize_activated_dimensions,
            {"model": model, "dataloader": dataloader, "device": device,
             "text": f"{name} Activated Dimensions"},
            "activated_dimensions.png"
        )
    ]
    for viz_func, kwargs, filename in graphics:
        if VERBOSE:
            print(f"\t- {viz_func}")
        fig = viz_func(**kwargs)
        if to_browser:
            fig.show()
        else:
            fig.write_image(os.path.join(PATH_GRAPHICS, name, filename))


def main() -> None:
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.split.lower() == "test":
        dataloader = get_test_dataloader(args.dataset)
    else:
        dataloader = get_train_dataloader(args.dataset)

    os.makedirs(PATH_GRAPHICS, exist_ok=True)
    models_to_visualize = [
        ("autoencoder", Autoencoder(), PATH_AE),
        ("vae", VAE(), PATH_VAE),
        ("vsc", VSC(), PATH_VSC),
        ("vsc_warmup", VSC(), PATH_VSC_WARMUP)
    ]
    selected = [m.strip().lower() for m in args.models.split(",")]
    for name, model_class, model_path in models_to_visualize:
        if name not in selected:
            continue
        model = load_model(model_class, model_path, device)
        if model is None:
            print(f"Model {name} not found")
            continue
        generate_graphics(model, name, dataloader,
                                device, args.to_browser)


if __name__ == "__main__":
    main()
