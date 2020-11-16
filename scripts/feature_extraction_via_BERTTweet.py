import os 
import glob 

import multiprocessing

import numpy as np 
import pandas as pd 

import torch
from transformers import AutoTokenizer
from transformers import AutoModel, AutoTokenizer


import utils_


ncores = multiprocessing.cpu_count() - 1


def get_BERTTweet_features(tweet):
    pre_trained_model = "vinai/bertweet-covid19-base-cased" #  "vinai/bertweet-base" # "bertweet-covid19-base-cased"
    bertweet = AutoModel.from_pretrained(pre_trained_model)
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    # Load the AutoTokenizer with a normalization mode if the input Tweet is raw
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, normalization=True)

    # from transformers import BertweetTokenizer
    # tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

    # line = "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"

    input_ids = torch.tensor([tokenizer.encode(tweet)])


    with torch.no_grad():
        _, feautre_2D = bertweet(input_ids)  # Models outputs are now tuples

    return feautre_2D.numpy().ravel().tolist()


# def get_features(df, tweet_txt_col_idx):
#     n_tweets = df.shape[0]
#     L = []
#     for i in range(n_tweets):
#         print("=" * 80 )
#         print("Tweet no.:", i + 1)
#         # print("Tweet:", df.iloc[i, tweet_txt_col_idx])
#         feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
#         L.append(feature)

#     return pd.DataFrame(L, columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]))])


def get_features(df, tweet_txt_col_idx):
    n_tweets = df.shape[0]
    L = []
    for i in range(n_tweets):
        print("=" * 80 )
        print("Tweet no.:", i + 1)
        # print("Tweet:", df.iloc[i, tweet_txt_col_idx])
        while utils_.is_internet_available() is False:
        	print("No internet connect")
        else:
        	feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
        L.append(feature)

    if len(L) != n_tweets:
        print("Something went wrong.")


    return pd.DataFrame(L, columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]))])



def wrt_single_file(filename, wrt_dir):
    df = pd.read_csv(filename)
    latent_val_df = get_features(df, 0)

    # write file to disk  
    latent_filename = filename.split("/")[-1]
    latent_val_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)

    return 0 


def wrt_feature_files(data_dir, wrt_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))[1:]
    print(files)

    tweet_txt_col_idx = 0 # column no 0 is the tweet text column 

    for file in files:
        print("Filename:", file)
        df = pd.read_csv(file)
        print(f'Number of tweets {df.shape[0]}')

        # try:
        latent_val_df = get_features(df, tweet_txt_col_idx)

        # write file to disk  
        latent_filename = file.split("/")[-1]
        latent_val_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)
        # except ValueError:
        #     print("Could not processed the file.")
        #     print("Filename with ValueError:", file)

    return 0 


if __name__ == "__main__":
    data_dir = "../data/nomask_tweets_v2_eng"
    wrt_dir = "../data/BERTTweet_AntiMask_Features"

    _ = wrt_feature_files(data_dir, wrt_dir)


    filename = "August_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    filename = "July_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    filename = "June_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    filename = "March_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    filename = "November_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)



    filename = "October_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    filename = "April_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)



    # promask tweets 
    data_dir = "../data/promask_tweets_eng"
    wrt_dir = "../data/BERTWeet_ProMask_Features"


    filename = "April_NoMask_Tweets.csv"
    wrt_dir = "../data/BERTTweet_Features"

    wrt_single_file(os.path.join("../data/promask_tweets_eng", filename), wrt_dir)



    