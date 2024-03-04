import pandas as pd
import csv
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def run_linear(df, attention_score_columns, bio_feature_columns):
    results = {}
    for layer_head in attention_score_columns:
        
        X = None
        Y = None
        # subset on bio columns


        # loop over rows (sequences)
        for index, seq in df.iterrows():
            # for each row, take values as numpy array, convert to matrix (# kmers x features)
            x = np.concatenate(seq[bio_feature_columns].values).ravel().reshape((44, 198)).T
            y = seq[layer_head]

            if X is not None:
                X = np.concatenate([X, x])
                Y = np.concatenate([Y, y])
            else:
                X = x
                Y = y

        X = np.nan_to_num(X)
        
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        model = LinearRegression().fit(X_train, y_train)
        
        #predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Coefficient of Determination (R^2): {r2}')
        
        #store coefficients and metrics
        results[layer_head] = {'coefficients': model.coef_, 'mse': mse, 'r2': r2}

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/scores/DNABERT_kmer_scores.csv",
        type=str,
        help="The path to the data",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/",
        type=str,
        help="The full path to the collect_coef.py file",
    )
    parser.add_argument(
        "--model_name",
        default="DNABERT",
        type=str,
        help="The model being explained",
    )
    args = parser.parse_args()

    data_path = args.data_path
    full_path = args.full_path
    model = args.model_name
    
    df = pd.read_csv(data_path, sep=';')
    #convert the string representations of lists in the columns (except 'Sequence' and 'kmer') into numpy arrays
    for col in df.columns:
        if col not in ['Sequence', 'kmer']:
            # convert the comma-separated string values to lists, then to numpy arrays
            df[col] = df[col].apply(lambda x: np.array(x.split(','), dtype=float))
    
    attention_score_columns = [col for col in df.columns if ('layer' in col and 'head' in col)]
    bio_feature_columns = [col for col in df.columns if not (('layer' in col and 'head' in col) or ('Sequence' in col) or ('kmer' in col))]

    results = run_linear(df, attention_score_columns, bio_feature_columns)

    dict_temp = {}
    for key in results.keys():
        dict_temp[key] = results[key]['coefficients'].tolist() + [results[key]['mse'], results[key]['r2']]

    df_results = pd.DataFrame.from_dict(dict_temp, orient="index", columns=bio_feature_columns + ["mse", "r2"])

    df_results.to_csv(f'{full_path}/data/coef/{model}_results_full.csv')


if __name__ == "__main__":
    main()