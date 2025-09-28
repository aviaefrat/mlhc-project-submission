import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import duckdb

from . import utils

nltk.download("punkt")
nltk.download('punkt_tab')

logger = utils.make_logger(name='embeddings_features', filename='embeddings_features.log')


def clean_text(text: str) -> str:
  if not isinstance(text, str):
    return ""
  # remove PHI placeholders like [**...**]
  text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
  text = text.lower().strip()
  text = ''.join(ch if ch.isalpha() or ch.isspace() else ' ' for ch in text)
  return text


def note_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def aggregate_embeddings(group):
    arrs = np.vstack(group['w2v_emb'].values)
    return arrs.mean(axis=0)


def build_embeddings_features(cohort_path: Path = None, db_path: Path = None):
    notes_sql = f"""
                SELECT
                    noteevents.subject_id::INTEGER AS subject_id,
                    noteevents.hadm_id::INTEGER AS hadm_id,
                    noteevents.charttime::TIMESTAMP AS charttime,
                    noteevents.category,
                    noteevents.text,
                    admissions.admittime::TIMESTAMP AS admittime
                FROM
                    NOTEEVENTS noteevents
                INNER JOIN admissions
                    ON noteevents.subject_id = admissions.subject_id
                      AND noteevents.hadm_id = admissions.hadm_id
                WHERE
                    noteevents.charttime::TIMESTAMP BETWEEN admissions.admittime::TIMESTAMP
                        AND (admissions.admittime::TIMESTAMP + INTERVAL '48 hours')
                  -- Exclude notes that are identified as errors
                  AND noteevents.iserror IS DISTINCT FROM 1;
                """
    db_path = db_path or utils.DB_PATH
    con = duckdb.connect(str(db_path))
    notes_df = con.execute(notes_sql).fetchdf().rename(str.lower, axis='columns')

    cohort_path = cohort_path or utils.FILTERED_COHORT_PATH
    cohort = pd.read_parquet(cohort_path)
    cohort_ids = cohort[['subject_id', 'admission_id']]

    # keep admissions according to our cohort
    notes_df = notes_df.rename(columns={"hadm_id": "admission_id"})
    notes_cohort = notes_df.merge(cohort_ids, on=['subject_id', 'admission_id'], how="inner")
    notes_cohort = notes_cohort[notes_cohort['category'].isin(
        ["Nursing", "Nursing/other", "Physician"])]  # After inspection, we settled on these categories.
    notes_cohort['clean_text'] = notes_cohort['text'].apply(clean_text)
    # Tokenize notes
    notes_cohort['tokens'] = notes_cohort['clean_text'].apply(lambda x: word_tokenize(x))

    # Train Word2Vec model
    logger.info(f"Creating Word2Vec embeddings")
    w2v_model = Word2Vec(
        sentences=notes_cohort['tokens'],
        vector_size=200,
        window=5,
        min_count=5,
        workers=os.cpu_count() - 1
    )

    notes_cohort['w2v_emb'] = notes_cohort['tokens'].apply(lambda x: note_vector(x, w2v_model))

    # Aggregate by 'subject_id', 'admission_id' - mean w2v for all notes
    logger.info(f"Creating computing mean Word2Vec embeddings for each subject")
    notes_cohort_w2v = notes_cohort.drop(columns=['category', 'text', 'clean_text', 'tokens'])
    agg_w2v = (
        notes_cohort_w2v
        .groupby(['subject_id', 'admission_id'])
        .apply(aggregate_embeddings)
        .reset_index()
    )

    embeddings_expanded = pd.DataFrame(agg_w2v[0].tolist(), index=agg_w2v.index)
    embeddings_expanded.columns = [f'emb_{i}' for i in range(embeddings_expanded.shape[1])]
    notes_features = pd.concat([agg_w2v[['subject_id', 'admission_id']], embeddings_expanded], axis=1)

    # This `.merge` will add NaNs to ids with no notes
    notes_features = cohort.merge(notes_features, on=['subject_id', 'admission_id'], how='left')

    # save
    utils.EMBEDDINGS_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    notes_features.to_parquet(utils.EMBEDDINGS_FEATURES_PATH, index=False)
    logger.info(f"Wrote embeddings features to: {utils.EMBEDDINGS_FEATURES_PATH}")

if __name__ == "__main__":
    build_embeddings_features(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)