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

    return feautre_2D#.numpy().ravel().tolist()


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


# def get_features(df, tweet_txt_col_idx):
#     n_tweets = df.shape[0]
#     L = []
#     for i in range(n_tweets):
#         print("=" * 80 )
#         print("Tweet no.:", i + 1)
#         # print("Tweet:", df.iloc[i, tweet_txt_col_idx])
#         # while utils_.is_internet_available() is False:
#         #     print("No internet connect")
#         # else:
#         #     feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
#         # L.append(feature)

#         if utils_.is_internet_available():
#             feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
#             L.append(feature)

#     if len(L) != n_tweets:
#         print("Something went wrong.")


#     return pd.DataFrame(L, columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]))])


# def get_features(df, tweet_txt_col_idx):
#     n_tweets = df.shape[0]
#     L = []

#     while len(L) < n_tweets:
#         try: 
#             feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
#         except ValueError:



#     for i in range(n_tweets):
#         print("=" * 80 )
#         print("Tweet no.:", i + 1)

#         if utils_.is_internet_available():
#             feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
#             L.append(feature)

#     if len(L) != n_tweets:
#         print("Something went wrong.")


#     return pd.DataFrame(L, columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]))])



def get_features_v1(df, tweet_txt_col_idx, row_idx, flag=False):
    n_tweets = df.shape[0]
    local_L = []
    for i in range(row_idx, n_tweets):
        if i == 10:
            flag = True
            return i, local_L
        else:
            local_L.append(i)

        # try:
        #     feature = get_BERTTweet_features(df.iloc[i, tweet_txt_col_idx])
        #     local_L.append(feature)
        # except ValueError:
        #     flag = True 
        #     return i, local_L


def get_features(df, tweet_txt_col_idx):
    L = []
    i = 0 
    n_tweets = df.shape[0]
    flag = False 
    while len(L) <= n_tweets:
        # if flag == False:
        #     local_L = get_features_v1(df, tweet_txt_col_idx, i)
        if flag == True:            
            i, local_L = get_features_v1(df, tweet_txt_col_idx, i)
            next_ = i 
            local_L = get_features_v1(df, tweet_txt_col_idx, next_)
            

        else:
            local_L = get_features_v1(df, tweet_txt_col_idx, i)

        i += 1 

        L.extend(local_L)

    return L # pd.DataFrame(L, columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]))])


def wrt_single_file(filename, wrt_dir):
    df = pd.read_csv(filename)
    latent_val_df = get_features(df, 0)

    # write file to disk  
    latent_filename = filename.split("/")[-1]
    latent_val_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)

    return 0 

def wrt_single_file(filename, wrt_dir):
    df = pd.read_csv(filename)
    # latent_val_df = get_features(df, 0)

    # # write file to disk  
    # latent_filename = filename.split("/")[-1]
    # latent_val_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)

    return df



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


def wrt_BERT_features_toFile(raw_filename, bert_filename):
    raw_df = pd.read_csv(raw_filename) 
    bert_df = pd.read_csv(bert_filename)
    print(raw_df.shape)
    print(bert_df.shape)

    left = bert_df.shape[0]
    right = raw_df.shape[0]

    tweet_txt_col_idx = 0 #
    L = []
    for j in range(left, 10):
        print(j, right)
        try:
            latent_val_df = get_BERTTweet_features(raw_df.iloc[j, tweet_txt_col_idx]) 
            # print(latent_val_df.shape)

            # bert_df.append(pd.DataFrame(latent_val_df.numpy(), columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]))]), ignore_index=True)
            # print("Number of tweets processed so far:", bert_df.shape)
            print([raw_df["ID"][j]])
            L.append(latent_val_df.numpy().ravel().tolist() + [raw_df["ID"][j]]) # attach tweet ID to latent vector 
            print("Length of list:", len(L))

            # bert_filename = bert_filename
            # bert_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)
            # return bert_df, bert_df.shape
        except ValueError:
            break 
            # bert_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)
            # return bert_df, j 

    bert_df = bert_df.append(pd.DataFrame(L, columns=["BERTWeet_" + str((i + 1)) for i in range(len(L[0]) - 1)] + ["ID"]))
    print("Dimension of bert processed df:", bert_df.shape)
    bert_df.to_csv(bert_filename, index=False, header=True)

    return bert_df


# def wrt_BERT_features_toFile_dirs(raw_data_dir, bert_data_dir):
#     raw_files = glob.glob(os.path.join(raw_data_dir, "*.csv")) # [1:]
#     bert_files = glob.glob(os.path.join(bert_data_dir, "*.csv"))
#     print(files)

#     tweet_txt_col_idx = 0 # column no 0 is the tweet text column 

#     for i in range(len(raw_files)):
#         if raw_files[i] == bert_files[i]:
#             raw_df = pd.read_csv(raw_files[i]) 
#             bert_df = pd.read_csv(bert_files[i])

#             for j in range(bert_df.shape[0], raw_df.shape[1]):
#                 try:
#                     latent_val_df = get_features(df, tweet_txt_col_idx) 

#                     bert_df.append(pd.DataFrame(latent_val_df.numpy()))

#                     bert_filename = bert_files.split("/")[-1]
#                     bert_df.to_csv(os.path.join(wrt_dir, latent_filename), index=False, header=True)
#                 except ValueError:
#                     return bert_df, j 


if __name__ == "__main__":
    data_dir = "../data/nomask_tweets_v2_eng"
    wrt_dir = "../data/BERTTweet_AntiMask_Features"

    # _ = wrt_feature_files(data_dir, wrt_dir)


    # filename = "August_NoMask_Tweets.csv"
    # wrt_dir = "../data/BERTTweet_Features"

    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    # filename = "July_NoMask_Tweets.csv"
    # wrt_dir = "../data/BERTTweet_Features"

    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    # filename = "June_NoMask_Tweets.csv"
    # wrt_dir = "../data/BERTTweet_Features"

    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    # filename = "March_NoMask_Tweets.csv"
    # wrt_dir = "../data/BERTTweet_Features"

    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    # filename = "November_NoMask_Tweets.csv"
    # wrt_dir = "../data/BERTTweet_Features"

    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)



    # filename = "October_NoMask_Tweets.csv"
    # wrt_dir = "../data/BERTTweet_Features"

    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    # filename = "April_NoMask_Tweets.csv"
    # wrt_dir = "../data/BERTTweet_Features"

    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)



    # # promask tweets 
    # data_dir = "../data/promask_tweets_eng"
    # bert_dir = "../data/BERTWeet_ProMask_Features"



    # # _ = wrt_feature_files(data_dir, wrt_dir)

    # filename = "April_WearMask_Tweets.csv"
    # # # wrt_dir = "../data/BERTWeet_ProMask_Features"

    # # wrt_single_file(os.path.join("../data/promask_tweets_eng", filename), wrt_dir)

    # df = wrt_BERT_features_toFile(os.path.join(data_dir, filename), os.path.join(bert_dir, filename))



    # # filename = "July_WearMask_Tweets.csv"
    # # wrt_dir = "../data/BERTWeet_ProMask_Features"

    # # wrt_single_file(os.path.join("../data/promask_tweets_eng", filename), wrt_dir)


    # # # AntiMask tweets 
    # data_dir = "../data/nomask_tweets_v2_eng"
    # bert_antiMask_dir = "../data/BERTTweet_AntiMask_Features"

    # # _ = wrt_feature_files(data_dir, wrt_dir)


    # filename = "April_NoMask_Tweets.csv"
    # # wrt_dir = "../data/BERTTweet_AntiMask_Features"

    # df = wrt_BERT_features_toFile(os.path.join(data_dir, filename), os.path.join(bert_antiMask_dir, filename))
    # # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)


    data_dir = "../data/stack_files"
    bert_antiMask_dir = "../data/stack_files"

    # # _ = wrt_feature_files(data_dir, wrt_dir)


    filename = "balanced_pro_n_anti_mask_df.csv"
    BERT_filename = "balanced_pro_n_anti_mask_BERT_df.csv"
    # wrt_dir = "../data/BERTTweet_AntiMask_Features"

    df = wrt_BERT_features_toFile(os.path.join(data_dir, filename), os.path.join(bert_antiMask_dir, BERT_filename))
    # wrt_single_file(os.path.join("../data/nomask_tweets_v2_eng", filename), wrt_dir)