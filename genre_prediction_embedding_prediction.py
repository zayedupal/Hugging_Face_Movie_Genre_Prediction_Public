import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm


def get_pipeline(index):
    pipelines = [
        # solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
        [('classifier', LogisticRegression(solver='saga', verbose=1))], # 0.485975
        # [('normalizer',Normalizer()), ('classifier', LogisticRegression(solver='saga', verbose=1))], # 0.485975
        # [('classifier',  GaussianProcessClassifier(1.0 * RBF(1.0)))],
        [('classifier', MLPClassifier(solver='adam', hidden_layer_sizes=(512, 256, 128, 64,),
                                                      early_stopping=True, random_state=1, verbose=1))],
        [('classifier', RandomForestClassifier(verbose=1))],
        # ('classifier', SVC(gamma='auto')),
        # ('classifier', DecisionTreeClassifier(random_state=0)),
    ]
    pipe = Pipeline(pipelines[index])

    return pipe


def generate_submission(data, model, output_path):
    ids = data['id']
    y_pred = model.predict(data['embeddings'].tolist())
    submission_df = pd.DataFrame()
    submission_df['id'] = ids
    submission_df['genre'] = y_pred
    submission_df.to_csv(output_path, index=False)


def remove_duplicates_by_relevancy(relevant_train_id_csv, train_df):
    relevant_id_df = pd.read_csv(relevant_train_id_csv)
    filtered_df = pd.merge(relevant_id_df, train_df, how="inner", on="id").reset_index(drop=True)
    filtered_df.drop(columns=["Unnamed: 0"], inplace=True)
    print(f'filtered_df: {len(filtered_df)}')
    return filtered_df


def train_evaluate_model(model_pipes, train_eval_dfs, embedding_csvs, split=0.7, hard_voting=False, print_accuracy=False):
    print(f'split: {split}')
    eval_dfs = []
    for model_pipe, train_eval_df in zip(model_pipes, train_eval_dfs):
        if not split:
            model_pipe.fit(train_eval_df['embeddings'].tolist(), train_eval_df['label'])
            # return None
        else:
            # break down train and eval
            train_df = train_eval_df[:int(len(train_eval_df) * split)]
            eval_df = train_eval_df[int(len(train_eval_df) * split):]
            eval_dfs.append(eval_df)

            # fit the data
            model_pipe.fit(train_df['embeddings'].tolist(), train_df['label'])

            # y_pred_eval = model_pipe.predict(eval_df['embeddings'].tolist())
            #
            # accuracy = accuracy_score(eval_df['label'], y_pred_eval)
            # print("Accuracy per model:", accuracy)

            calculate_combined_pred(test_dfs=eval_dfs, models=model_pipes, embedding_csvs=embedding_csvs,
                                    hard_voting=hard_voting, print_accuracy=print_accuracy)


def calculate_combined_pred(test_dfs, models, embedding_csvs=[], hard_voting=False, print_accuracy=False):
    print(f'calculate_combined_pred: {print_accuracy}')
    np_result = None
    classes = None

    if hard_voting:
        for i, (test_df, model_pipe) in enumerate(zip(test_dfs, models)):
            pred_proba = model_pipe.predict_proba(test_df["embeddings"].tolist())
            classes = model_pipe.classes_
            tolerance = 1e-6
            if not i:
                np_result = np.array([[1 if (max(row) == item) else 0 for item in row] for row in pred_proba])
                # print(f'np_result: {np_result}')
            else:
                clamped_pred = np.array([[1 if (max(row) == item) else 0 for item in row] for row in pred_proba])
                np_result += clamped_pred

            if print_accuracy:
                accuracy = accuracy_score(test_df['label'], classes[np.argmax(pred_proba, axis=1)])
                print(f"Accuracy for model #{i}: {accuracy}")

        max_indices = np.argmax(np_result, axis=1)
    else:
        for i, (test_df, model_pipe, embedding_csv) in enumerate(zip(test_dfs, models, embedding_csvs)):
            regex = r"embeddings/(.*?)\.csv"
            model_pred_file_name = re.sub(regex, r"\1", embedding_csv['train'])
            pred_proba = model_pipe.predict_proba(test_df["embeddings"].tolist())
            print(f'model_pred_file_name: {model_pred_file_name}')
            print(f'cur model classes: {model_pipe.classes_}')

            classes = model_pipe.classes_
            if not i:
                np_result = pred_proba
            else:
                np_result += pred_proba

            print(f'print_accuracy: {print_accuracy}')
            if print_accuracy:
                if not os.path.exists(f"pred_proba/pred_proba_{model_pred_file_name}.csv"):
                    pred_proba_df = pd.DataFrame({'pred_proba': pred_proba.tolist()})
                    pred_proba_df.to_csv(f"pred_proba/pred_proba_{model_pred_file_name}.csv")

                accuracy = accuracy_score(test_df['label'], classes[np.argmax(pred_proba, axis=1)])
                print(f"Accuracy for model #{i}: {accuracy}")

        np_result = np_result / len(models)
        max_indices = np.argmax(np_result, axis=1)

    # print(f'max_indices: {max_indices}')
    y_pred = classes[max_indices]
    if print_accuracy:
        accuracy = accuracy_score(test_dfs[0]['label'], y_pred)
        print("Accuracy combined:", accuracy)

    return y_pred

def generate_submission_combined(test_dfs, models, embedding_csvs, submission_file, hard_voting=False):
    y_pred = calculate_combined_pred(test_dfs=test_dfs, models=models, embedding_csvs=embedding_csvs, hard_voting=hard_voting)

    ids = test_dfs[0]['id']
    submission_df = pd.DataFrame()
    submission_df['id'] = ids
    submission_df['genre'] = y_pred
    submission_df.to_csv(submission_file, index=False)

def convert_string_to_float_array_efficient(string):
    """Converts a string like "[1.23, 4.56, 7.89]" to a float array."""

    return [float(number) for number in string[1:-1].split(", ")]

if __name__ == '__main__':
    start_time = time.time()
    tqdm.pandas()

    embedding_csvs = [
        # {
        #     "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_intfloat_multilingual-e5-large.csv",
        #     "test": "embeddings/test_embeddings_intfloat_multilingual-e5-large.csv",
        #     "cleanup": False,
        #     # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_intfloat_multilingual-e5-large.csv"
        # },
        # {
        #     "train": "embeddings/train_cleaned_embeddings_intfloat_e5-large-v2.csv",
        #     "test": "embeddings/test_embeddings_intfloat_e5-large-v2.csv",
        #     "cleanup": False
        # },
        {
            "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_sentence-transformers_sentence-t5-xxl.csv",
            "test": "embeddings/test_embeddings_sentence-transformers_sentence-t5-xxl.csv",
            "cleanup": False,
            # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_sentence-transformers_sentence-t5-xxl.csv"
        },
        # {
        #     "train": "embeddings/train_cleaned_embeddings_sentence-transformers_sentence-t5-xl.csv",
        #     "test": "embeddings/test_embeddings_sentence-transformers_sentence-t5-xl.csv",
        #     "cleanup": False,
        #     # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_sentence-transformers_sentence-t5-xxl.csv"
        # },
        # {
        #     "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_hkunlp_instructor-xl.csv",
        #     "test": "embeddings/test_embeddings_hkunlp_instructor-xl.csv",
        #     "cleanup": False,
        #     # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_hkunlp_instructor-xl.csv"
        # },
        # {
        #     "train": "embeddings/train_cleaned_embeddings_sentence-transformers_gtr-t5-xxl.csv",
        #     "test": "embeddings/test_embeddings_sentence-transformers_gtr-t5-xxl.csv",
        #     "cleanup": False
        # },
        # {
        #     "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_google_mt5-xl.csv",
        #     "test": "embeddings/test_embeddings_google_mt5-xl.csv",
        #     "cleanup": False,
        #     # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_google_mt5-xl.csv"
        # },
        {
            "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_google_flan-t5-xl.csv",
            "test": "embeddings/test_embeddings_google_flan-t5-xl.csv",
            "cleanup": False,
            # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_google_flan-t5-xl.csv"
        },
        {
            "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_google_flan-t5-xxl.csv",
            "test": "embeddings/test_embeddings_google_flan-t5-xxl.csv",
            "cleanup": False,
            # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_google_flan-t5-xxl.csv"
        },
        # {
        #     "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_microsoft_deberta-v2-xxlarge.csv",
        #     "test": "embeddings/test_embeddings_microsoft_deberta-v2-xxlarge.csv",
        #     "cleanup": False,
        #     # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_microsoft_deberta-v2-xxlarge.csv"
        # },
        # {
        #     "train": "embeddings/train_cleaned_by_t5-xxl_embeddings_thenlper_gte-large.csv",
        #     "test": "embeddings/test_embeddings_thenlper_gte-large.csv",
        #     "cleanup": False,
        #     # "cleanup_csv": "embeddings/train_cleaned_by_t5-xxl_embeddings_thenlper_gte-large.csv"
        # }
    ]

    models = []
    test_dfs = []
    train_eval_dfs = []

    # select which pipeline to use
    pipeline_indexes = [0]
    for i, entry in enumerate(embedding_csvs):
        train_embedding_csv, test_embedding_csv, cleanup = entry["train"], entry["test"], entry["cleanup"]

        # get embeddings
        train_eval_df = pd.read_csv(train_embedding_csv)
        test_df = pd.read_csv(test_embedding_csv)
        print(len(train_eval_df), len(test_df))
        # continue
        # cleanup
        if cleanup:
            cleanup_csv = entry['cleanup_csv']
            if entry["train"] == cleanup_csv:
                print("Error: train cleaned will be overwritten")
                exit()
            train_eval_df = remove_duplicates_by_relevancy("data/train_cleaned_ids_t5_xxl.csv", train_eval_df)
            train_eval_df.to_csv(cleanup_csv, index=False)

        # shorter dataset
        # train_eval_df = train_eval_df[:2000]
        # test_df = test_df[:2000]

        train_eval_df['embeddings'] = train_eval_df["embeddings"].progress_apply(
            lambda x: convert_string_to_float_array_efficient(x)
        )
        test_df['embeddings'] = test_df["embeddings"].progress_apply(
            lambda x: convert_string_to_float_array_efficient(x)
        )
        print(f"embedding cleanup time: {time.time() - start_time}")

        for pipe in pipeline_indexes:
            model_pipe = get_pipeline(pipe)

            models.append(model_pipe)
            test_dfs.append(test_df)
            train_eval_dfs.append(train_eval_df)

    # train on partial data and evaluate model, if split is none train on full data
    # train_evaluate_model(models, train_eval_dfs, embedding_csvs, split=0.7, print_accuracy=True, hard_voting=False)
    train_evaluate_model(models, train_eval_dfs, embedding_csvs, split=None)

    # generate_submission_combined(test_dfs, models, embedding_csvs, 'submissions/submission_cleaned_full_gte-large_LR-saga.csv')
    generate_submission_combined(test_dfs, models, embedding_csvs, 'submissions/submission_cleaned_full_t5-xxl_flan-t5-xl_flan-t5-xxl_LR-saga.csv')
    # generate_submission_combined(test_dfs, models, 'submissions/submission_cleaned_full_meta-llama_Llama-2-7b-hf_LR-saga.csv')

    print(f"Runtime: {time.time() - start_time}")
