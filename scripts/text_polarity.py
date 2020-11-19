import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import utils_ 


def get_polarity(df):
    sentiment_model = SentimentIntensityAnalyzer()
    df['sentimentDict'] = df['Tweet Text'].apply(lambda tweet: sentiment_model.polarity_scores(tweet))
    
    df['vader_neg']  = df['sentimentDict'].apply(lambda score_dict: score_dict['neg'])
    df['vader_neu']  = df['sentimentDict'].apply(lambda score_dict: score_dict['neu'])
    df['vader_pos']  = df['sentimentDict'].apply(lambda score_dict: score_dict['pos'])
    df['vader_compound']  = df['sentimentDict'].apply(lambda score_dict: score_dict['compound'])


    return df.drop(columns=["sentimentDict"]) # df[["vader_neg", "vader_neu", "vader_pos", "vader_compound"]] # df.drop(columns=["sentimentDict"])


if __name__=="__main__":
    # df = pd.read_csv("../data/stack_files/samp_raw_df_april.csv")
    # df2 = get_polarity(df)
    # df2.head()

    data_dir = "../data/stack_files/"
    filename = "balanced_pro_n_anti_mask_df.csv"

    df = pd.read_csv(os.path.join(data_dir, filename))


    # include fold information 

    df["folds"] = utils_.get_folds(df)
    df2 = get_polarity(df)
    
    filename = "balanced_pro_n_anti_mask_df_v2.csv"
    df2.to_csv(os.path.join(wrt_dir, filename), index=False, header=True)



    # dummy dataset 
    data_dir = "../data/stack_files/"
    filename = "samp_raw_df_april.csv"

    df = pd.read_csv(os.path.join(data_dir, filename))


    # include fold information 

    df["folds"] = utils_.get_folds(df)
    df2 = get_polarity(df)

    filename = "samp_raw_df_april_v2.csv"
    df2.to_csv(os.path.join(wrt_dir, filename), index=False, header=True)
