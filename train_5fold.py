import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from dataset import Prediction_Dataset, Pretrain_Collater, Finetune_Collater
from metrics import AverageMeter, Records_R2, Records_AUC
import os
from model import PredictionModel, BertModel
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='data/train_data.txt',
                    help='Path to txt file with SMILES and labels')
parser.add_argument('--task-name', type=str, default='classification', help='Name of the classification task')
parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--no-pretrain', action='store_true', help='Do not use pretrained weights')
parser.add_argument('--pretrained-path', type=str, default='weights/medium_weights_bert_encoder_weightsmedium_40.pt',
                    help='Path to pretrained weights')
args = parser.parse_args()


def upsample_positive_class(df, target_col='label', random_state=None):
    positive_samples = df[df[target_col] == 1].copy()
    negative_samples = df[df[target_col] == 0].copy()

    print(f"Before upsampling - Positive: {len(positive_samples)}, Negative: {len(negative_samples)}")

    neg_count = len(negative_samples)
    pos_count = len(positive_samples)

    if pos_count == 0:
        print("Warning: No positive samples found!")
        return df

    if pos_count >= neg_count:
        print("Positive class already equal or larger than negative class, no upsampling needed.")
        return df

    upsample_ratio = neg_count // pos_count
    remaining_samples = neg_count % pos_count

    upsampled_positive = []

    for _ in range(upsample_ratio):
        upsampled_positive.append(positive_samples.copy())

    if remaining_samples > 0:
        if random_state is not None:
            remaining_pos = positive_samples.sample(n=remaining_samples, random_state=random_state)
        else:
            remaining_pos = positive_samples.sample(n=remaining_samples)
        upsampled_positive.append(remaining_pos)

    if upsampled_positive:
        all_positive = pd.concat(upsampled_positive, ignore_index=True)
    else:
        all_positive = positive_samples

    balanced_df = pd.concat([all_positive, negative_samples], ignore_index=True)

    if random_state is not None:
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    print(f"After upsampling - Positive: {len(all_positive)}, Negative: {len(negative_samples)}")
    print(f"Total samples: {len(balanced_df)}")

    return balanced_df


class MetricsCalculator:

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def update(self, y_prob, y_true):

        for i in range(y_prob.shape[0]):
            if y_true[i, 0] != -1000:
                self.y_prob.append(y_prob[i, 0])
                self.y_true.append(y_true[i, 0])
                self.y_pred.append(1 if y_prob[i, 0] > 0.5 else 0)

    def calculate_metrics(self):
        if len(self.y_true) == 0:
            return {'acc': 0, 'sen': 0, 'spe': 0, 'auc': 0}

        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_prob = np.array(self.y_prob)


        acc = accuracy_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sen = tp / (tp + fn) if (tp + fn) > 0 else 0

        spe = tn / (tn + fp) if (tn + fp) > 0 else 0

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        return {'acc': acc, 'sen': sen, 'spe': spe, 'auc': auc}


def load_data_from_txt(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                smiles = parts[0]
                try:
                    label = int(parts[1])
                    data.append({'SMILES': smiles, 'label': label})
                except ValueError:
                    print(f"Skipping invalid line: {line}")
                    continue

    df = pd.DataFrame(data)
    print(f"Successfully loaded {len(df)} valid samples")
    return df


def main():
    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 2, 'd_model': 128, 'path': 'small_weights'}
    medium = {'name': 'Medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights'}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 512, 'path': 'large_weights'}

    arch = medium
    pretraining = not args.no_pretrain

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    dff = d_model * 4
    vocab_size = 60
    dropout_rate = 0.3

    seed = 123
    print(f"\n{'=' * 80}")
    print(f"Running with SEED {seed}")
    print(f"{'=' * 80}")

    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Loading training data from {args.data_path}")
    df = load_data_from_txt(args.data_path)
    print(f"Loaded {len(df)} samples")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=seed)

    fold_results = {
        'acc': [],
        'sen': [],
        'spe': [],
        'auc': []
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print(f"\nFold {fold + 1}/{args.n_folds}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(f"Original train size: {len(train_df)}, Val size: {len(val_df)}")

        train_df_upsampled = upsample_positive_class(train_df, target_col='label', random_state=seed)

        print(f"Final train size after upsampling: {len(train_df_upsampled)}")

        train_dataset = Prediction_Dataset(train_df_upsampled, smiles_head='SMILES',
                                           reg_heads=[], clf_heads=['label'])
        val_dataset = Prediction_Dataset(val_df, smiles_head='SMILES',
                                         reg_heads=[], clf_heads=['label'])

        collator_args = argparse.Namespace(clf_heads=['label'], reg_heads=[])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=Finetune_Collater(collator_args))
        val_dataloader = DataLoader(val_dataset, batch_size=128,
                                    shuffle=False, collate_fn=Finetune_Collater(collator_args))

        model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff,
                                num_heads=num_heads, vocab_size=vocab_size,
                                dropout_rate=dropout_rate, reg_nums=0, clf_nums=1)


        if pretraining and args.pretrained_path and os.path.exists(args.pretrained_path):
            model.encoder.load_state_dict(torch.load(args.pretrained_path, map_location=device))
            print(f'Loaded pretrained weights from {args.pretrained_path}')

        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

        train_loss = AverageMeter()
        val_loss = AverageMeter()

        train_metrics = MetricsCalculator()
        val_metrics = MetricsCalculator()

        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

        def train_step(x, properties):
            model.train()
            clf_true = properties['clf']
            properties_pred = model(x)
            clf_pred = properties_pred['clf']

            loss = (loss_func(clf_pred, clf_true * (clf_true != -1000).float()) *
                    (clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum() + 1e-6)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics.update(torch.sigmoid(clf_pred).detach().cpu().numpy(),
                                 clf_true.detach().cpu().numpy())
            train_loss.update(loss.detach().cpu().item(), x.shape[0])

        def eval_step(x, properties, metrics_calc, metrics_loss):
            model.eval()
            with torch.no_grad():
                clf_true = properties['clf']
                properties_pred = model(x)
                clf_pred = properties_pred['clf']

                loss = (loss_func(clf_pred, clf_true * (clf_true != -1000).float()) *
                        (clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum() + 1e-6)

                metrics_calc.update(torch.sigmoid(clf_pred).detach().cpu().numpy(),
                                    clf_true.detach().cpu().numpy())
                metrics_loss.update(loss.detach().cpu().item(), x.shape[0])

        for epoch in range(args.epochs):
            train_metrics.reset()
            train_loss.reset()
            for x, properties in train_dataloader:
                train_step(x, properties)

            train_results = train_metrics.calculate_metrics()

            val_metrics.reset()
            val_loss.reset()
            for x, properties in val_dataloader:
                eval_step(x, properties, val_metrics, val_loss)

            val_results = val_metrics.calculate_metrics()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{args.epochs}:')
                print(f'  Train - AUC: {train_results["auc"]:.4f}, ACC: {train_results["acc"]:.4f}, '
                      f'SEN: {train_results["sen"]:.4f}, SPE: {train_results["spe"]:.4f}')
                print(f'  Valid - AUC: {val_results["auc"]:.4f}, ACC: {val_results["acc"]:.4f}, '
                      f'SEN: {val_results["sen"]:.4f}, SPE: {val_results["spe"]:.4f}')

        final_val_metrics = val_results

        print(f"\nFold {fold + 1} Final Validation Metrics (Epoch {args.epochs}):")
        print(f"  AUC: {final_val_metrics['auc']:.4f}, ACC: {final_val_metrics['acc']:.4f}, "
              f"SEN: {final_val_metrics['sen']:.4f}, SPE: {final_val_metrics['spe']:.4f}")

        fold_results['acc'].append(final_val_metrics['acc'])
        fold_results['sen'].append(final_val_metrics['sen'])
        fold_results['spe'].append(final_val_metrics['spe'])
        fold_results['auc'].append(final_val_metrics['auc'])

    print(f"\n{'=' * 80}")
    print("5-FOLD CROSS-VALIDATION RESULTS (MEAN ± STD) - FINAL EPOCH")
    print(f"{'=' * 80}")

    for metric in ['acc', 'sen', 'spe', 'auc']:
        values = np.array(fold_results[metric])
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

    results_summary = {
        'metric': ['ACC', 'SEN', 'SPE', 'AUC'],
        'mean': [],
        'std': [],
        'fold_values': []
    }

    for metric in ['acc', 'sen', 'spe', 'auc']:
        values = np.array(fold_results[metric])
        results_summary['mean'].append(np.mean(values))
        results_summary['std'].append(np.std(values))
        results_summary['fold_values'].append(values.tolist())

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('5fold_cv_results.csv', index=False)
    print(f"\nDetailed results saved to 5fold_cv_results.csv")

    fold_df = pd.DataFrame(fold_results)
    fold_df['fold'] = range(1, 6)
    fold_df = fold_df[['fold', 'acc', 'sen', 'spe', 'auc']]
    fold_df.to_csv('individual_fold_results.csv', index=False)
    print(f"Individual fold results saved to individual_fold_results.csv")


if __name__ == '__main__':
    main()
