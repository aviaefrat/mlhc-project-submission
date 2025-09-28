# Machine Learning for Healthcare – Final Project (Spring 2025)
Avia Efrat and Or Honovich

## Repository Overview
There are 3 main top-level directories:
 - `data`, where all the data we need and create is stored.
 - `src`, for all the source files (except unseen_data_evaluation.py).
 - `reports`, for storing and loading experiment results (models, metrics, etc.)

## Our Data Pipeline
### Conventions
Our codebase uses predetermined paths for accessing data, for ease of use.  
All these paths are defined in `src/utils.py` in UPPERCASE, e.g. `utils.INITIAL_COHORT_CSV`, and are referenced throughout this README. Note that sometime we use shorthands, e.g. `INITIAL_COHORT_CSV` instead of `utils.INITIAL_COHORT_CSV`.
You *can* specify other values for these paths, as demonstrated in `unseen_data_evaluation.py`.

### Cohort Filtering
From initial_cohort.csv, we create our filtered cohort using src/patient_timeline_filtering.py.
To do this, we run the following code:
``` python
    initial_subject_ids = get_subject_ids_from_cohort_csv_path(cohort_csv_path=utils.INITIAL_COHORT_CSV)  # For our model training we used the cohort csv utils.INITIAL_COHORT_CSV. If you want to start from another cohort, just change the value of the cohort_csv_path argument to the path of your initial cohort csv (e.g. the notebook uses test_example.csv as its initial cohort csv).
    
    create_first_admissions_df(subject_ids=initial_subject_ids, db_path=utils.DB_PATH)  # creates FIRST_ADMISSIONS_PATH
    
    create_min_54h_first_admissions_df(db_path=utils.DB_PATH) # creates MIN_54H_FIRST_ADMISSIONS_PATH (from FIRST_ADMISSIONS_PATH)
   
    create_min_54h_first_admissions_age_filtered_df(db_path=utils.DB_PATH)  # creates MIN_54H_FIRST_ADMISSIONS_FILTERED_AGE_PATH (from MIN_54H_FIRST_ADMISSIONS_PATH). Notice that in src/utils.py, we define `FILTERED_COHORT_PATH = MIN_54H_FIRST_ADMISSIONS_FILTERED_AGE_PATH`. Below you can see that we create all the features based on the parquet saved at FILTERED_COHORT_PATH.
    
    create_second_admission_df(db_path=utils.DB_PATH)  # creates SECOND_ADMISSIONS_PATH, which is a helper dataframe needed for creating the readmission label (see below)
```
Exploratory data analysis can be found at `src/MLHC_EDA.ipynb`

### Labels
To create the labels, we run the following code from `src/labels.py`:
``` python
    create_mortality_df(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)  # creates MORTALITY_LABEL_PATH
    create_prolonged_stay_df(cohort_path=utils.FILTERED_COHORT_PATH)  # creates PROLONGED_STAY_LABEL_PATH
    create_readmission_df(cohort_path=utils.FILTERED_COHORT_PATH, second_admissions_path=utils.SECOND_ADMISSIONS_PATH)  # creates READMISSION_LABEL_PATH
```

### Feature Creation
Exploratory 
For most if the features, we separated the feature creation into two steps to make analysis easier.
#### Demographics
`src/demographic_features.py`:
``` python
    create_demographics_df(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)  # creates DEMOGRAPHICS_PATH
```
#### Vitals
The first step is to create helper dataframes using `src/vitals.py`:
``` python
    create_all_vitals_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)  # creates VITALS_48H_LABS_PATH
    base_vitals_df = pd.read_parquet(utils.FIRST_48H_VITALS_PATH)
    create_vitals_df(base_vitals_df)  # creates VITALS_PATH
```
Then we use `src/vitals_features.py`:
``` python
    build_vitals_features(cohort_path=utils.FILTERED_COHORT_PATH)  # creates VITALS_FEATURES_PATH
```

#### Labs
The first step is to create helper dataframes using `src/labs.py`:
``` python
    create_all_labs_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)  # creates FIRST_48H_LABS_PATH
    base_labs_df = pd.read_parquet(utils.FIRST_48H_LABS_PATH)
    create_labs_df(base_labs_df)  # creates LABS_PATH
```
Then we use `src/labs_features.py`:
``` python
    build_labs_features(cohort_path=utils.FILTERED_COHORT_PATH)  # creates LABS_FEATURES_PATH
```

#### GCS (Glasgow Coma Scale)
The first step is to create helper dataframes using `src/gcs.py`:
``` python
    create_all_gcs_for_filtered_cohort_0_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)  # creates FIRST_48H_GCS_PATH
    base_gcs_df = pd.read_parquet(utils.FIRST_48H_GCS_PATH)
    create_gcs_df(base_gcs_df)  # creates GCS_PATH
```
Then we use `src/gcs_features.py`:
``` python
    build_gcs_features(cohort_path=utils.FILTERED_COHORT_PATH)  # creates GCS_FEATURES_PATH
```

#### Prescriptions
The first step is to create helper dataframes using `src/prescriptions.py`:
``` python
    extract_prescriptions_first_48h(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)  # creates FIRST_48H_PRESCRIPTIONS_PATH
```
Then we use `src/prescriptions_features.py`:
``` python
    build_prescriptions_features(cohort_path=utils.FILTERED_COHORT_PATH)  # creates PRESCRIPTIONS_FEATURES_PATH
```

#### Noteevents .TEXT Embeddings
We don't create any helper dataframes here, and directly create the embedding features using `src/embeddings_features.py`
``` python
    build_embeddings_features(cohort_path=utils.FILTERED_COHORT_PATH, db_path=utils.DB_PATH)  # creates EMBEDDINGS_FEATURES_PATH
```

### Training
The code for training is in `src/train.py` and the `src/modeling` directory.
The Entry point for training is `training.main()`.
Via the CLI, you specify the target, what data you want to exclude from feature creation (e.g. `--no-demographics`, `--no-labs`), various hyperparameters for the model (and if you want to perform a grid search), and other arguments like `--val-size` and `seed`.

The results and artifacts of the training are written to `--save-dir`.  
Important artifacts include the model (`model_xgb.joblib`), the calibrator (if exists, `calibrator.joblib`), and the preprocessor (`preprocessor.joblib`), so we can apply transformations fitted on the train set on a new cohort if needed.
