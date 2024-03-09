import os
import argparse

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_path))

from datasets.cifar10 import get_cifar10
from models.cifar10 import Cifar10Model, resnet18
from core.defenses.sabre import SabreWrapper
from experiments.utils import ceil_dec
from experiments.setup import setup_attacks, run_exp


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Run CIFAR-10 experiments with specified model and attacks')
    parser.add_argument('--model_name', type=str, default='SABRE',
                        choices=['baseline', 'Adversarial Training', 'SABRE'],
                        help='The name of the model to run the experiment on.')
    parser.add_argument('--epsilon', type=float, default=8. / 255,
                        help='The perturbation limit for the adversarial attack.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The batch size for training and testing.')
    parser.add_argument('--n_variants', type=int, default=10,
                        help='The number of random variants used by SABRE.')
    parser.add_argument('--normalize', action='store_true',
                        help='Flag to enable the normalization of samples before classification.')
    parser.add_argument('--disable_rand', action='store_true',
                        help='Flag to disable the use of randomness in SABRE.')

    args = parser.parse_args()

    # Use args to access command-line arguments
    model_name = args.model_name
    epsilon = ceil_dec(args.epsilon, 3)
    batch_size = args.batch_size
    n_variants = args.n_variants
    normalize = args.normalize
    use_rand = not args.disable_rand

    dataset_name = "cifar10"
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load CIFAR-10 Data
    data_root = os.path.join(root_path, "datasets")
    train_data_loader = get_cifar10(batch_size=batch_size, data_root=data_root, train=True, val=False,
                                    return_loader=True)
    test_data_loader = get_cifar10(batch_size=batch_size, data_root=data_root, train=False, val=True,
                                   return_loader=True)

    # Model Training Parameters
    n_channels = 3
    learning_rate = 1e-2
    train_params = {
        "baseline": {"n_channels": n_channels, "normalize": normalize, "learning_rate": learning_rate},
        "Adversarial Training": {"eps": epsilon, "alpha": ceil_dec(2. / 255), "steps": 7, "n_channels": n_channels,
                                 "normalize": normalize, "learning_rate": learning_rate},
        "SABRE": {"eps": epsilon, "n_channels": n_channels, "normalize": normalize, "learning_rate": learning_rate}
    }

    # Initialize Models
    baseline_model = resnet18()
    sabre_model = SabreWrapper(eps=epsilon, use_rand=use_rand, n_variants=n_variants, base_model=Cifar10Model())
    models = {"baseline": baseline_model, "Adversarial Training": baseline_model, "SABRE": sabre_model}

    # Setup Attacks
    attacks = setup_attacks(eps=epsilon, bounds=(0, 1), n_classes=len(classes))

    # Run Experiment
    run_exp(model_name=model_name,
            dataset_name=dataset_name,
            model=models[model_name],
            attacks=attacks,
            eps=epsilon,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            train_params=train_params[model_name])


if __name__ == "__main__":
    main()
