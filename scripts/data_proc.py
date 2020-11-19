import os 
import glob 

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split



import utils_
import eda 
import text_polarity


def stack_files(data_dir, wrt_dir, mask_cat):
    files = glob.glob(os.path.join(data_dir, "*.csv"))

    # total_num_eng_tweets = 0 
    # filename_n_nTweets = {}
    new_added_cols = 8
    n_cols = 16 + new_added_cols 
    mat  = np.zeros((0, n_cols))
    for file in files:
        print("Filename:", file)
        df = pd.read_csv(file)
        print(f'Number of tweets {df.shape}')
        print(df.columns)

        df_eng = eda.tweet_lang(df)

        month = file.split("/")[-1].split("_")[0]
        # if month == "March":
        #     df_eng["User Location"] = -1

        df_eng["month"] = [month] * df_eng.shape[0]
        df_eng["ground_truth"] = [mask_cat] * df_eng.shape[0]

        id_name = str(mask_cat) + "_" + month
        length = df_eng.shape[0]
        df_eng["CAP_6317_sample_ID"] = index_generator(id_name, length)

        # # folds ID 
        # df_eng["folds"] = utils_.get_folds(df_eng)

        # polarity_df = text_polarity.get_polarity(df_eng)

        # df_eng["vader_neg"] = polarity_df["vader_neg"].tolist()
        # df_eng["vader_neu"] = polarity_df["vader_neu"].tolist()
        # df_eng["vader_pos"] = polarity_df["vader_pos"].tolist()
        # df_eng["vader_compound"] = polarity_df["vader_compound"].tolist()

        print("English", df_eng.shape)

        mat = np.append(mat, df_eng.values, axis=0)

        # # write the file to the dir 
        # eng_filename = file.split("/")[-1]
        # df_eng.to_csv(os.path.join(wrt_dir, eng_filename), index=False, header=True)

    # print("Total tweets:", total_num_eng_tweets)
    # print("Filename and no of tweets:", filename_n_nTweets)
    col_names = ['Tweet Text', 'Tweet Datetime', 'Tweet Id', 'User Id', 'User Name', 'User Location', 'Tweet Coordinates', 
                'Place Info', 'Country', 'Hashtags', 'Retweets', 'Favorites', 'Language', 'Source', 'Replied Tweet Id', 
                'Replied Tweet User Id', 'month', 'ground_truth', 'ID', "folds", "vader_neg", "vader_neu", "vader_pos",
                 "vader_compound"]


    return pd.DataFrame(mat, columns=col_names)


def index_generator(filename, length):
    max_digits = 7
    indices = []
    for idx in range(1, length + 1):
        digits_in_i = len(str(idx))

        if digits_in_i == max_digits:
            index = filename + "_" + str(i)
        else:
            padding_len = max_digits - digits_in_i
            index = filename + "_" + "0" * padding_len + str(idx)
        indices.append(index)

    return np.expand_dims(indices, axis=1)



def get_sample_proMask_tweets(df):
    size_ = round(8158 / 38855, 2)
    samp_df, _, _, _ = train_test_split(df, df["ground_truth"], test_size=1 - size_, random_state=2020, stratify=df["month"])

    return samp_df


def merge_proMask_w_antiMask(antiMask_df, proMask_df):

    return pd.concat([antiMask_df, proMask_df], ignore_index=True)


def remove_inconsistent_rows(df):
	df = df[df["ground_truth"].isin(["0", "1"]) == True]
	df = df[df["folds"].isin([["1", "2", "3", "4"]]) == True]
	df =df[df["Favorites"].isin(["en"]) == False]

	if df["month"].unique().tolist() == 9 and df["ID"].unique().tolist() == df.shape[0]:
		return df


if __name__=="__main__":
    # process promask tweets 
    data_dir = "../data/WearMask_Tweets"
    wrt_dir = "../data/stack_files"

    pro_mask = 1 
    proMask_df = stack_files(data_dir, wrt_dir, pro_mask) 

    filename = "ProMask_raw_data.csv"
    proMask_df.to_csv(os.path.join(wrt_dir, filename), index=False, header=True)




    data_dir = "../data/NoMask_Tweets_Version_2"
    wrt_dir = "../data/stack_files"

    mask = 0
    antiMask_df = stack_files(data_dir, wrt_dir, mask) 

    filename = "AntiMask_raw_data.csv"
    antiMask_df.to_csv(os.path.join(wrt_dir, filename), index=False, header=True)



    # balanced merge antiMask and proMask datas 
    m_df = merge_proMask_w_antiMask(antiMask_df, get_sample_proMask_tweets(proMask_df))

    filename = "balanced_pro_n_anti_mask_df.csv"
    m_df.to_csv(os.path.join(wrt_dir, filename), index=False, header=True)



    # merge antiMask and proMask datas 
    m_df = merge_proMask_w_antiMask(antiMask_df, proMask_df)

    filename = "pro_n_anti_mask_df.csv"
    m_df.to_csv(os.path.join(wrt_dir, filename), index=False, header=True)


    # remove rows 
    # filename = "balanced_pro_n_anti_mask_df_v2.csv"
    # data_dir = "../data/stack_files"

    # df = pd.read_csv(os.path.join(data_dir, filename))

    # df = remove_inconsistent_rows(df)

    # filename = "balanced_pro_n_anti_mask_df_v3.csv"
    # df.to_csv(os.path.join(data_dir, filename), index=False, header=True)




    # subsample 20% tweet keeping fold balanced 
    filename = "balanced_pro_n_anti_mask_df_v3.csv"
    df = pd.read_csv(os.path.join("../data/stack_files", filename))

    seed_ = 2020
    samp_df, _, _, _ = train_test_split(df, df["ground_truth"], 
	                                    test_size=0.8, # 0.3, 0.95
	                                    random_state=seed_, 
	                                    stratify=df["folds"])

    filename = "balanced_pro_n_anti_mask_df_v4.csv"
    samp_df.to_csv(os.path.join("../data/stack_files", filename), index=False, header=True)
