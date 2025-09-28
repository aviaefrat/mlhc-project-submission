import pandas as pd
from sklearn.model_selection import train_test_split

from .. import utils

LOGGER = utils.make_logger("split", "split.log")


def split_train_val_test(df: pd.DataFrame, ycol: str, test_size: float, val_size: float, seed: int):
    """
    Takes df that is assumed to contain id columns (subject_id and admission_id), all the feature columns, and a label column.
    Returns X_train, X_val, X_test, y_train, y_val, y_test
    """
    y = df[ycol].astype(int).values
    X = df.drop(columns=[ycol])

    X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed, stratify=y
    )
    test_size_relatively_to_val_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_and_test, y_val_and_test, test_size=test_size_relatively_to_val_size, random_state=seed, stratify=y_val_and_test
    )

    LOGGER.info(f"Split sizes – train: {len(X_train):,}, val: {len(X_val):,}, test: {len(X_test):,}")
    for set_name, set_label_values in [("train", y_train), ("val", y_val), ("test", y_test)]:
        label_value_counts_in_set = {int(k): int(v) for k, v in pd.Series(set_label_values).value_counts().sort_index().items()}
        LOGGER.info(f"Label counts in {set_name}: {label_value_counts_in_set}")

    return X_train, X_val, X_test, y_train, y_val, y_test
