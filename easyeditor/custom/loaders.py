import pandas as pd

from ast import literal_eval

def load_data(data_path = "../catco-data/"):

    data_dict = dict()

    types_df = pd.read_csv(data_path + "animal-type-tokens.tsv", sep="\t")
    properties_df = pd.read_csv(data_path + "animal-data.tsv", sep="\t")

    edits_df = pd.read_csv(data_path + "edits.csv")
    baseline_df = pd.read_csv(data_path + "baseline-evaluation.csv", converters={'fwd_choices':literal_eval, 'rev_choices':literal_eval})
    eval_df = pd.read_csv( data_path + "edits-evaluation.csv", converters={'fwd_choices':literal_eval, 'rev_choices':literal_eval})

    return baseline_df, edits_df, eval_df


import configparser

def auth_token():

    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["hugging_face"]["token"]


def load_prefixes(prefix_lines = -1, verbose=False):
    with open('prefix_fwd.txt') as f:
        prefix_fwd = "".join(f.readlines()[0:prefix_lines])

    with open('prefix_rev.txt') as f:
        prefix_rev = "".join(f.readlines()[0:prefix_lines])

    # prefix_fwd = f.read()
    if verbose:
        print(prefix_fwd)
        print("---")
        print(prefix_rev)
        print("---")

    return prefix_fwd, prefix_rev