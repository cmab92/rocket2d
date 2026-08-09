"""Command-line interface for rocket2d.

Subcommands mirror the stages of the original notebook:

    rocket2d download                 # fetch & normalize Kaggle datasets
    rocket2d inspect neu              # print stats / show samples for a dataset
    rocket2d train cnn neu            # train+evaluate the CNN baseline
    rocket2d train rocket neu         # train+evaluate the ROCKET classifier
    rocket2d train all                # run CNN and ROCKET on neu, xray, dtd
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from rocket2d.config import Config
from rocket2d.data.datasets import ImageDataset
from rocket2d.data.kaggle import download_and_prepare_datasets, print_directory_tree
from rocket2d.training import run_cnn, run_rocket
from rocket2d.visualization import grayscale_histogram, image_size_stats, show_samples

DATASETS = ["neu", "xray", "dtd"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocket2d", description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root directory for datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download and normalize Kaggle datasets.")

    inspect_p = subparsers.add_parser(
        "inspect", help="Print stats and sample images for a dataset."
    )
    inspect_p.add_argument("dataset", choices=DATASETS)
    inspect_p.add_argument("--no-show", action="store_true", help="Skip interactive plt.show().")

    train_p = subparsers.add_parser("train", help="Train and evaluate a model.")
    train_sub = train_p.add_subparsers(dest="model", required=True)

    cnn_p = train_sub.add_parser("cnn", help="Train the SimpleCNN baseline.")
    cnn_p.add_argument("dataset", choices=DATASETS)
    cnn_p.add_argument("--image-size", type=int, default=128)
    cnn_p.add_argument("--batch-size", type=int, default=32)
    cnn_p.add_argument("--epochs", type=int, default=10)
    cnn_p.add_argument("--device", default=None, help="Defaults to CUDA if available, else CPU.")
    cnn_p.add_argument("--no-show", action="store_true")

    rocket_p = train_sub.add_parser("rocket", help="Train the ROCKET + RidgeClassifier pipeline.")
    rocket_p.add_argument("dataset", choices=DATASETS)
    rocket_p.add_argument("--image-size", type=int, default=128)
    rocket_p.add_argument("--n-kernels", type=int, default=5000)
    rocket_p.add_argument(
        "--device", default=None, help="Defaults to CUDA if available, else CPU."
    )
    rocket_p.add_argument("--no-show", action="store_true")

    all_p = train_sub.add_parser("all", help="Run CNN and ROCKET on every dataset.")
    all_p.add_argument("--image-size", type=int, default=128)
    all_p.add_argument("--batch-size", type=int, default=32)
    all_p.add_argument("--epochs", type=int, default=10)
    all_p.add_argument("--n-kernels", type=int, default=5000)
    all_p.add_argument("--device", default=None)
    all_p.add_argument("--no-show", action="store_true")

    return parser


def _cmd_download(config: Config) -> None:
    download_and_prepare_datasets(config.base_data_dir)
    print_directory_tree(config.base_data_dir)


def _cmd_inspect(config: Config, dataset: str, show: bool) -> None:
    path = config.dataset_dirs[dataset]
    print("\n" + "=" * 50)
    print(f"DATASET: {dataset}")
    print("=" * 50)

    ds = ImageDataset(path, dataset=dataset)
    idx_to_class = {idx: name for name, idx in ds.class_to_idx.items()}
    images = [sample_path for sample_path, _, _ in ds.samples]
    labels = [idx_to_class[label_idx] for _, label_idx, _ in ds.samples]
    print(f"Total images: {len(images)}")
    print(f"Number of classes: {len(ds.class_to_idx)}")

    print("\nClass distribution (top 10):")
    for label, count in Counter(labels).most_common(10):
        print(f"{label}: {count}")

    print("\nSample images:")
    show_samples(path, grayscale=True, show=show)

    print("\nImage size stats:")
    image_size_stats(images)

    print("\nGrayscale histograms:")
    grayscale_histogram(images, show=show)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``rocket2d`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config(seed=args.seed, base_data_dir=args.data_dir)

    if args.command == "download":
        _cmd_download(config)
    elif args.command == "inspect":
        _cmd_inspect(config, args.dataset, show=not args.no_show)
    elif args.command == "train":
        if args.model == "cnn":
            run_cnn(
                args.dataset,
                config.dataset_dirs[args.dataset],
                img_size=args.image_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                seed=config.seed,
                device=args.device or config.device,
                show_plots=not args.no_show,
            )
        elif args.model == "rocket":
            run_rocket(
                args.dataset,
                config.dataset_dirs[args.dataset],
                img_size=args.image_size,
                seed=config.seed,
                n_kernels=args.n_kernels,
                device=args.device or config.device,
                show_plots=not args.no_show,
            )
        elif args.model == "all":
            device = args.device or config.device
            for name in DATASETS:
                print(f"\n{'=' * 30}\nRUN CNN: {name.upper()}\n{'=' * 30}")
                run_cnn(
                    name,
                    config.dataset_dirs[name],
                    img_size=args.image_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    seed=config.seed,
                    device=device,
                    show_plots=not args.no_show,
                )
            for name in DATASETS:
                print(f"\n{'=' * 30}\nRUN ROCKET: {name.upper()}\n{'=' * 30}")
                run_rocket(
                    name,
                    config.dataset_dirs[name],
                    img_size=args.image_size,
                    seed=config.seed,
                    n_kernels=args.n_kernels,
                    device=device,
                    show_plots=not args.no_show,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
