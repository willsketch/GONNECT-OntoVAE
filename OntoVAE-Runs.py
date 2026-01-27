#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from onto_vae.ontobj import Ontobj
from onto_vae.vae_model import OntoVAE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


# --------------------------
# Utility functions
# --------------------------

def load_split(expr_path, split_ids_path, nan_cols):
    """
    Load a subset of expression data given patient IDs and drop metadata columns.
    """
    df = pd.read_csv(expr_path)
    ids = np.load(split_ids_path, allow_pickle=True)
    df = df[df["patient_id"].isin(ids)]
    df = df.dropna(subset='cancer_type')
    labels = df['cancer_type'].copy()
    df = df.drop(columns=nan_cols)
    df = df.set_index('patient_id', drop=True)
    df.index.name = None
    df = df.T
    df.columns.name = None
    return df, labels

#TODO create variable ontology description
def setup_ontology(obo_path, gene_annot_path, top_thresh=1000, bottom_thresh=30, description='GO-untrimmed'):
    """
    Initialize, trim, and create masks for Ontobj.
    """
    ont = Ontobj(description=description)
    ont.initialize_dag(obo=obo_path, gene_annot=gene_annot_path)
    ont.trim_dag(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    ont.create_masks(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    return ont


def match_dataset(ontobj, expr_df, name, top_thresh=1000, bottom_thresh=30):
    """
    Match the dataframe to the ontology within Ontobj.
    """
    ontobj.match_dataset(expr_data=expr_df, name=name, top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    return ontobj


def train_ontovae(ontobj, dataset_name, top_thresh=1000, bottom_thresh=30,
                  save_path='models/best_model.pt', lr=1e-4, kl_coeff=1e-4, batch_size=32, epochs=20, log=False):
    """
    Initialize OntoVAE and train the model.
    """
    model = OntoVAE(ontobj=ontobj, dataset=dataset_name, top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    model.to(model.device)


    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.train_model(save_path, lr=lr, kl_coeff=kl_coeff, batch_size=batch_size, epochs=epochs, log=log)
    return model


def compute_latent_embeddings(model, ontobj, dataset_name, top_thresh, bottom_thresh):
    """
    Return latent embeddings for a given dataset.
    """
    model.eval()
    device = model.device
    trim_key = f'{top_thresh}_{bottom_thresh}'

    if trim_key not in ontobj.data:
        raise KeyError(f'Trim key {trim_key} not found in {ontobj.dataset}. '
                       f'First set up ontology with [{top_thresh}, {bottom_thresh}]')
    elif dataset_name not in ontobj.data[trim_key]:
        raise KeyError(f'Dataset {dataset_name} not found in {ontobj.dataset}.'
                       f'Match dataset: {dataset_name} with onology object')
    elif ontobj.data[trim_key][dataset_name] is None:
        raise ValueError(f'Dataset in ontology object is None')
    else:
        x = torch.tensor(ontobj.data[trim_key][dataset_name], dtype=torch.float32, device=device)

    with torch.no_grad():
        latent_x = model.get_embedding(x).cpu().numpy()

    return latent_x

def mse_loss(model, ontObj, dataset_name, top_thresh, bottom_thresh):
    """
    Calculate mse of reconstruction for given dataset.
    """
    trim_key = f'{top_thresh}_{bottom_thresh}'

    rec_x = model.get_reconstructed_values(ontObj, dataset_name)
    rec_x = torch.tensor(rec_x, dtype=torch.float32)
    x = ontObj.data[trim_key][dataset_name]
    x = torch.tensor(x, dtype=torch.float32)
    mse = F.mse_loss(rec_x, x, reduction='mean')
    return mse.item()

def clustering_metrics(latent, labels, seed=42):
    """
    Compute clustering metrics: NMI, ARI, Silhouette Score.
    """
    n_clusters = labels.nunique()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_assignments = kmeans.fit_predict(latent)

    nmi = normalized_mutual_info_score(labels, cluster_assignments)
    ari = adjusted_rand_score(labels, cluster_assignments)

    # Silhouette requires >1 cluster
    sil_score = silhouette_score(latent, cluster_assignments) if n_clusters > 1 else np.nan

    return {'NMI': nmi, 'ARI': ari, 'Silhouette': sil_score}


# --------------------------
# Main pipeline per run
# --------------------------

def run_experiment(run_seed, expr_data_path, split_dir, nan_cols, base_ontObj, model_dir='models',
                   top_thresh_ontobj=1000, bottom_thresh_ontobj=30, batch_size=32, epochs=20,
                   lr=1e-4, kl_coeff=1e-4):
    """
    Run a full OntoVAE training + evaluation for a single seed.
    Saves the trained model and metrics to disk.
    """

    print(f'Running OntoVAE for split with seed: {run_seed}\n')

    #TODO: check if run folder and splits dir exists
    run_folder = os.path.join(split_dir, f'run_seed-{run_seed}')
    # os.makedirs(run_folder, exist_ok=True)

    # --------------------------
    # Load data
    # --------------------------
    train_path = os.path.join(run_folder, 'train_ids.npy')
    val_path = os.path.join(run_folder, 'val_ids.npy')
    test_path = os.path.join(run_folder, 'test_ids.npy')

    train_df, train_labels = load_split(expr_data_path, train_path, nan_cols)
    val_df, val_labels = load_split(expr_data_path, val_path, nan_cols)
    test_df, test_labels = load_split(expr_data_path, test_path, nan_cols)

    combined_train_df = pd.concat([train_df, val_df], axis=1)
    combined_train_labels = pd.concat([train_labels, val_labels], axis=0)

    #TODO also check whether labels still correspond to the patient id
    assert len(combined_train_df.columns) == len(combined_train_labels.index)

    print('Data loaded \n')
    # --------------------------
    # Match dataset
    # --------------------------
    #TODO make dataset name a variable?
    dataset_name_train = f'TCGA_run_seed-{run_seed}-train'
    dataset_name_test = f'TCGA_run_seed-{run_seed}-test'
    ont_train = match_dataset(base_ontObj, combined_train_df, name=dataset_name_train, top_thresh=top_thresh_ontobj,
                              bottom_thresh=bottom_thresh_ontobj)

    ont_test = match_dataset(base_ontObj, test_df, name=dataset_name_test, top_thresh=top_thresh_ontobj,
                             bottom_thresh=bottom_thresh_ontobj)

    print('Datasets matched with Ontology \n')
    # --------------------------
    # Train OntoVAE
    # --------------------------

    print('Starting training \n')
    model_dir_path = os.path.join(run_folder, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    model_save_path = os.path.join(model_dir_path, 'best_model.pt')
    model = train_ontovae(ont_train, dataset_name_train, top_thresh=top_thresh_ontobj, bottom_thresh=bottom_thresh_ontobj,
                          save_path=model_save_path, lr=lr, kl_coeff=kl_coeff, batch_size=batch_size, epochs=epochs)

    # --------------------------
    # Compute embeddings and metrics
    # --------------------------
    print('Computing metrics \n')

    mse_train = mse_loss(model, ont_train, dataset_name_train, top_thresh=top_thresh_ontobj,
                         bottom_thresh=bottom_thresh_ontobj)

    mse_test = mse_loss(model, ont_test, dataset_name_test, top_thresh=top_thresh_ontobj,
                        bottom_thresh=bottom_thresh_ontobj)

    latent_train = compute_latent_embeddings(model, ont_train, dataset_name_train, top_thresh=top_thresh_ontobj,
                                             bottom_thresh=bottom_thresh_ontobj)
    metrics_train = clustering_metrics(latent_train, combined_train_labels)
    metrics_train['mse'] = mse_train

    latent_test = compute_latent_embeddings(model, ont_test, dataset_name_test, top_thresh=top_thresh_ontobj,
                                            bottom_thresh=bottom_thresh_ontobj)
    metrics_test = clustering_metrics(latent_test, test_labels)
    metrics_test['mse'] = mse_test

    # --------------------------
    # Save metrics
    # --------------------------
    metrics = {
        'train': metrics_train,
        'test': metrics_test
    }

    metrics_path = os.path.join(split_dir, 'metrics.txt')
    with open(metrics_path, 'a') as f:
        f.write(f"run-{run_seed}: {metrics}\n")

    print(f"[Run_seed {run_seed}] Metrics saved to {metrics_path}")
    return metrics


# --------------------------
# Example main
# --------------------------
if __name__ == "__main__":
    # Paths
    expr_data_path = 'data/TCGA_complete_bp_top1k.csv'
    split_dir = 'splits'
    obo_path = 'data/go-basic.obo'
    gene_annot_path = 'data/gene_annot_ontovae.txt'
    nan_cols = ['sample_type', 'cancer_type', 'tumor_tissue_site', 'stage_pathologic_stage']
    ontology_description = 'GO-basic'
    model_dir = 'models'
    top_thresh_ontobj = 100
    bottom_thresh_ontobj = 30

    # Seeds to run
    seeds = [2, 3, 4, 5, 6]

    # --------------------------
    # Initialize ontology
    # --------------------------
    cached_ontObj_path = os.path.join(split_dir, 'cached_ontology.pkl')
    if os.path.exists(cached_ontObj_path):
        with open(cached_ontObj_path, "rb") as f:
            BaseOntObj = pickle.load(f)
        print("Loaded cached ontology")
    else:
        BaseOntObj = setup_ontology(obo_path, gene_annot_path, top_thresh=top_thresh_ontobj,
                                    bottom_thresh=bottom_thresh_ontobj,
                                    description=ontology_description)
        with open(cached_ontObj_path, "wb") as f:
            pickle.dump(BaseOntObj, f)
        print("Built and cached ontology")

    # Run all seeds
    for s in seeds:
        run_experiment(
            run_seed=s,
            expr_data_path=expr_data_path,
            split_dir=split_dir,
            nan_cols=nan_cols,
            base_ontObj=BaseOntObj,
            model_dir=model_dir,
            top_thresh_ontobj=top_thresh_ontobj,
            bottom_thresh_ontobj=bottom_thresh_ontobj,
            batch_size=2,
            epochs=1,
            lr=1e-4,
            kl_coeff=1e-4
        )
