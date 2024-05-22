import os
import json
import torch

from core.attacks import FGSM, PGD, EoTPGD, CW
from models.helpers import load_model
from experiments.utils import baseline_train, adversarial_train, sabre_train, evaluate
from utils.log import printer

from pathlib import Path
root_path = Path(__file__).resolve().parents[1]


def setup_attacks(eps, bounds, n_classes):
    attacks = {
        # FGSM attack
        'fgsm': FGSM(eps=eps, bounds=bounds),

        # PGD attacks
        'pgd_steps_7': PGD(eps=eps, alpha=2. * eps / 7, steps=7, bounds=bounds),
        'pgd_steps_20': PGD(eps=eps, alpha=2. * eps / 20, steps=20, bounds=bounds),
        'pgd_steps_100': PGD(eps=eps, alpha=2. * eps / 20, steps=100, bounds=bounds),
        'pgd_steps_1000': PGD(eps=eps, alpha=2. * eps / 20, steps=1000, bounds=bounds),

        # EoT-PGD attacks
        'eotpgd_steps_7_10': EoTPGD(eps=eps, alpha=2. * eps / 7, steps=7, eot_steps=10, bounds=bounds),
        'eotpgd_steps_20_10': EoTPGD(eps=eps, alpha=2. * eps / 20, steps=20, eot_steps=10, bounds=bounds),
        'eotpgd_steps_100_10': EoTPGD(eps=eps, alpha=2. * eps / 20, steps=100, eot_steps=10, bounds=bounds),
        'eotpgd_steps_1000_10': EoTPGD(eps=eps, alpha=2. * eps / 20, steps=1000, eot_steps=10, bounds=bounds),

        # Carlini-Wagner attack
        'cwl2': CW(c=1e-4, kappa=10, steps=1000, lr=1e-2, num_classes=n_classes, eps=eps, bounds=bounds),
    }
    return {
        '_': {
            'no_attack': lambda x, y: x,
        },
        'whitebox': attacks
    }


def create_directories(model_name, dataset_name):
    models_dir = os.path.join(root_path, "artifacts/saved_models", model_name, dataset_name)
    logs_dir = os.path.join(root_path, "artifacts/logs", model_name, dataset_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return models_dir, logs_dir


def select_training_function(model_name):
    if 'adversarial' in model_name.lower():
        return adversarial_train
    elif 'sabre' in model_name.lower():
        return sabre_train
    return baseline_train


def save_json(data, log_file):
    with open(log_file, "w") as f:
        json.dump(data, f)


def load_json(log_file):
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return json.load(f)
    return {}


def perform_attacks(model_name, model, attacks, train_fn, train_data_loader, test_data_loader,
                    models_dir, trials_records, log_file, traces_file, train_params):
    correct_counts = {f"{model_name}-{attack_type}_{attack_key}": 0
                      for attack_type in attacks.keys()
                      for attack_key in attacks[attack_type].keys()}
    total_counts = {f"{model_name}-{attack_type}_{attack_key}": 0
                    for attack_type in attacks.keys()
                    for attack_key in attacks[attack_type].keys()}

    for attack_type in attacks.keys():
        for attack_key in attacks[attack_type].keys():
            attack_method = attacks[attack_type][attack_key]

            key = f"{model_name}-{attack_type}_{attack_key}"

            if key in trials_records.keys():
                correct_counts[key] = trials_records[key]['correct_count']
                total_counts[key] = trials_records[key]['total_count']
                printer(f"Records found for {key}. Skipping.", traces_file)
                continue

            model_path = f"{models_dir}/model.weights.pth"

            if os.path.exists(model_path):
                printer(f'+Loading pretrained model...', traces_file)
            else:
                printer(f'\n+Training model...', traces_file)

                model = train_fn(model, train_data_loader, n_epochs=100, learning_rate=train_params["learning_rate"],
                                 save_path=model_path, log_path=traces_file, params=train_params)

            model = load_model(model, model_path=model_path, log_path=traces_file)

            printer(f'\nEvaluating model... [Attack={attack_type}/{attack_key}]', traces_file)

            correct, total = evaluate(model, attack_method, test_data_loader, log_path=traces_file)

            if key not in trials_records:
                trials_records[key] = {}
            trials_records[key]['correct_count'] = correct
            trials_records[key]['total_count'] = total
            trials_records[key]['model_path'] = model_path
            trials_records[key]['logs_path'] = traces_file

            correct_counts[key] = correct
            total_counts[key] = total

            save_json(trials_records, log_file)

    printer(f"{'=' * 100}", traces_file)


def compile_evaluation_results(trials_records):
    traces = f'\nEvaluation results...\n'
    traces += ('=' * 100) + '\n'

    for key in trials_records.keys():
        total_count = trials_records[key]['total_count']
        correct_count = trials_records[key]['correct_count']
        accuracy = 100. * correct_count / total_count

        traces += f'{key} | Accuracy: {accuracy:.2f}% ({correct_count}/{total_count}))\n'

    traces += ('=' * 100) + '\n'

    return traces


def run_exp(model_name, dataset_name, model, attacks, eps, train_data_loader, test_data_loader, train_params):
    models_dir, logs_dir = create_directories(model_name, dataset_name)
    records_file = os.path.join(logs_dir, "records.json")
    traces_file = os.path.join(logs_dir, "traces.txt")
    results_file = os.path.join(logs_dir, "results.txt")

    train_fn = select_training_function(model_name)
    trials_records = load_json(records_file)

    evaluation_results = '\n' + ('-' * 150) + '\n'
    evaluation_results += f"\nExperiment Results: {model_name}/{dataset_name}\n"
    evaluation_results += ('=' * 100) + '\n'

    printer(f'\nRunning Experiment: {model_name}/{dataset_name} | EPS={eps:.3f}...', traces_file)
    printer(f'GPU count: {torch.cuda.device_count()}\n', traces_file)

    printer(f'\nParameters: {train_params}\n\n', traces_file)

    perform_attacks(model_name, model, attacks, train_fn, train_data_loader, test_data_loader,
                    models_dir, trials_records, records_file, traces_file, train_params)

    save_json(trials_records, records_file)

    traces = compile_evaluation_results(trials_records)

    printer(traces, results_file)

    evaluation_results += traces
    printer(evaluation_results, traces_file)
