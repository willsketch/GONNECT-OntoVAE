import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, split=0.7, seed=1):
    validation_test_split = 0.5

    gene_expression = data.copy()

    train_set, remaining_set = train_test_split(
        gene_expression,
        train_size=split,
        random_state=seed
    )

    validation_set, test_set = train_test_split(
        remaining_set,
        train_size=validation_test_split,
        random_state=seed
    )

    return train_set, validation_set, test_set

if __name__ == "__main__":

    DATA_PATH = os.path.join(os.getcwd(), 'data/TCGA_complete_bp_top1k.csv')
    SEEDS = [2, 3, 4, 5, 6]
    OUTPUT_DIR = os.path.join(os.getcwd(), 'splits')
    expr_data_df = pd.read_csv(DATA_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_DIR, f"run_seed-{seed}")
        os.makedirs(run_dir, exist_ok=True)

        train, val, test = split_data(expr_data_df, seed=seed)

        train_path = os.path.join(run_dir, 'train_ids.npy')
        val_path = os.path.join(run_dir, 'val_ids.npy')
        test_path = os.path.join(run_dir, 'test_ids.npy')

        np.save(train_path, train["patient_id"].values)
        np.save(val_path, val["patient_id"].values)
        np.save(test_path, test["patient_id"].values)

        print(f"Run-seed-{seed}: "
              f"train={len(train)}, val={len(val)}, test={len(test)}")