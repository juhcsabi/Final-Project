import json
import os
from collections import defaultdict
import pandas as pd
from vaderSentiment import vaderSentiment
from plotly import graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import subreddits
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
json_path = os.path.join(data_path, "json")

def get_density(df, column):
    sns.kdeplot(df[column], fill=True)
    plt.title(f'Density Plot of {column}')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.show()

def overall_sentiment():
    avgs = defaultdict(list)
    avgs["mean"] = []
    df_ = pd.read_json(os.path.join(data_path, f"sentiment_evolutions.json"))
    df_.index = pd.DatetimeIndex(pd.date_range(pd.Timestamp("2012-01"), pd.Timestamp("2022-12"), freq="MS"))
    df_na = df_["mean"][df_["mean"].isna()]
    missing_months = df_na.index.values
    print(missing_months)
    for year in range(2012, 2023):
        for month in range(1, 13):
            if pd.Timestamp(f"{year}-{month}") not in missing_months:
                continue

            month_ = str(month).zfill(2)
            df = pd.DataFrame(columns=["id", "sentiment_score", "group"])
            print(os.path.join(data_path, f"sentiment/{year}-{month_}"))
            for root, dirs, files in os.walk(os.path.join(data_path, f"sentiment/{year}-{month_}")):
                for file in files:
                    print(file)
                    df_ym = pd.read_json(os.path.join(root, file))
                    df_ym = df_ym[df.columns]
                    df = pd.concat([df, df_ym])

            avgs["year"].append(year)
            avgs["month"].append(month)
            avgs["mean"].append(df["sentiment_score"].mean())
            avgs["median"].append(df["sentiment_score"].median())
            avgs["std"].append(df["sentiment_score"].std())
            for community in ["Republican", "Democrat"]:
                avgs[f"mean_{community}"].append(df[df["group"] == community]["sentiment_score"].mean())
                avgs[f"median_{community}"].append(df[df["group"] == community]["sentiment_score"].median())
                avgs[f"std_{community}"].append(df[df["group"] == community]["sentiment_score"].std())
    for i, missing_month in enumerate(missing_months):
        values = {key: avgs[key][i] for key in avgs}
        df_.loc[missing_month] = values
    print(df_.loc[missing_months])
    #values = ["0.03622002075875143", "0.018731146836732437", "-0.020166431686732704", "0.007076781344183981", "0.006668929900883526", "0.015194512917488122", "0.0034436162980552272", "0.013380051984953428", "0.026399346288903638", "0.015682986071719144", "0.03606547852072242", "-0.055076987350418896", "-0.052538391357081056", "-0.030156199194434276", "-0.01954622263889997", "-0.05060810378750614", "-0.025653207627510475", "-0.011798655196053763", "-0.02683553477444902", "-0.009398379638948476", "-0.02232243909602572", "0.013436767705887672", "0.011653032878244225", "0.016970638362109634", "0.011028188376880894", "0.0159100995925201", "0.026382258619122335", "0.023747244668932226", "0.007780991849088933", "-0.008361040603038111", "0.012906683646863929", "-0.031243822982088856", "0.012611013369942551", "0.006773089021511929", "0.006746274405085912", "-0.03834619302949062", "0.027219881529308813", "0.01336832770095592", "0.0238357826951306", "0.037646275956550866", "0.050866895031859344", "0.04729501007911036", "0.02872572565825052", "0.03457085865214021", "0.03997121472830207", "0.027958663001591878", "0.02393595942522153", "0.008103146891190885", "0.05616943895198955", "0.08607058907248476", "0.06201457821341397", "0.07655227249407207", "0.06208810050065893", "0.02346712503107764", "0.028552479318942687", "0.01752489249812371", "0.010084925417473493", "0.010055192880018193", "0.020875133561195865", "0.012876682128592398", "0.004001218515682414", "-0.00581170227353902", "0.008863100692391103", "0.006120772829113273", "0.00888044674025897", "0.012185925001732857", "0.01156367467870844", "-0.020449995631046877", "0.0051157271445109965", "-0.017204426522104282", "0.0014974184120517258", "0.010754979697455648", "0.012788446375420558", "-0.017333705688972516", "-0.0050097632225238236", "0.002251509057164535", "-0.003979335349434441", "-0.009438742432122973", "-0.0006015087641240974", "-0.0010361335241745706", "-0.013297368767862413", "-0.015755488805341163", "0.007546786338021029", "0.005459055919903045", "0.012031136651117068", "0.013227485113266269", "0.006115403521775719", "0.014261647241821359", "0.0038374525917062212", "0.011371621953497203", "-0.004835550338455434", "-0.00805843200663911", "0.011270914539211096", "0.02270175501422688", "0.02605520372951105", "0.030551025830095134", "0.01727529849060432", "0.06198980664171767", "0.0431379520576701", "0.013811452141442568", "-0.004703863710435107", "-7.61756035375649e-05", "0.004963997075104768", "0.013806551235722928", "0.013419096512387468", "0.03198783027226656", "0.053285580551979", "0.02967703271308114", "0.008575946651681725", "0.021992413369740622", "0.027700626099555013", "0.025623817413209304", "0.020789218497785294", "0.03115407064627374", "0.023379677953156822", "0.009143663165455496", "0.02480978887889026", "0.04053013117069485", "0.02649610223396796", "0.03919539447545353", "0.041496468010018224", "0.019795550820975682", "0.018585300456919557", "0.030879084005150666", "-0.003857053849600814", "0.009417937480220323", "0.017340408048858825", "0.02750598323178584", "0.029041852066825295", "0.02900963042333102", "0.04633268163463593", "0.03649532892991413"]
    #print(len(values))
    #print(len([f"{year}-{month}" for year in range(2012, 2023) for month in range(1, 13)]))
    figs = {
        "mean": go.Figure(),
        "median": go.Figure(),
        "std": go.Figure()
    }
    for key in avgs:
        for avg_type in figs:
            if avg_type in key:
                figs[avg_type].add_trace(go.Scatter(
                    x=[f"{year}-{month}" for year in range(2012, 2023) for month in range(1, 13)],
                    y=df_[key],
                    mode='lines+markers',
                    name=f'{key} sentiment'
                ))
    for avg_type in figs:
        figs[avg_type].update_layout(
        title='Vader-based Sentiment Evolution Over Time',
        width=1800,
        height=1000
        )

    # Show the figure
        figs[avg_type].show()
        figs[avg_type].write_html(os.path.join(data_path, f"{avg_type}_sentiment_evolution2.html"))

    df_.to_json(os.path.join(data_path, f"sentiment_evolutions2.json"))


def fill_sentiment(row):
    if pd.isna(row["sentiment_score"]):
        return vaderSentiment.SentimentIntensityAnalyzer().polarity_scores(row["body"])["compound"]
    return row["sentiment_score"]


def join_sentiment():
    for year in range(2012, 2023):
        for month in range(1, 13):
            month_str = f"0{month}" if month < 10 else str(month)
            df = pd.DataFrame(columns=["id", "sentiment_score"])
            for root, dirs, files in os.walk(os.path.join(data_path, f"thesis_results/{year}-{month}")):
                for file in files:
                    df_ym = pd.read_json(os.path.join(root, file), lines=True)
                    df = pd.concat([df, df_ym])
            for root, dirs, files in os.walk(os.path.join(json_path, f"{year}-{month_str}")):
                for file in files:
                    print(os.path.join(root, file))
                    df_large = pd.read_json(os.path.join(root, file))
                    df_large["group"] = df_large["subreddit"].str.lower().map(subreddits.subreddit_to_group)
                    df_large = df_large[df_large["group"].isin(["Democrat", "Republican"])]
                    df_large = df_large.merge(df, how="left", on="id")
                    df_large.fillna()


def update_vader_lexicon():
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    senticnet_df = pd.read_excel(os.path.join(data_path, "senticnet.xlsx"))
    print(senticnet_df)
    senticnet_df = senticnet_df[["CONCEPT", "POLARITY INTENSITY"]]
    i = 0
    for index, row in senticnet_df.iterrows():
        if i % 10000 == 0:
            print(i)
        sia.lexicon.update({row["CONCEPT"]: row["POLARITY INTENSITY"] * 4})
        i += 1
    with open(os.path.join(data_path, 'custom_vader_lexicon.json'), 'w') as f:
        json.dump(sia.lexicon, f)


def load_custom_lexicon():
    with open(os.path.join(data_path, 'custom_vader_lexicon.json'), 'r') as f:
        custom_lexicon = json.load(f)
    custom_sia = SentimentIntensityAnalyzer()
    custom_sia.lexicon.update(custom_lexicon)
    return custom_sia


def get_dirs():
    dirs = os.listdir(json_path)
    return [f"{json_path}/{dir}" for dir in dirs]


def vader_analysis():
    sia = load_custom_lexicon()
    dirs = get_dirs()
    print(dirs)
    for directory in dirs:
        os.makedirs(directory.replace("json", "sentiment"), exist_ok=True)
    for directory in dirs:
        for root, folders, files in os.walk(directory):
            if "2020-08" not in root and "2020-09" not in root:
                continue
            for file in files:
                if not file.endswith(".json"):
                    continue
                if os.path.exists(os.path.join(root.replace("json", "sentiment"), f"{file}")):
                    print("File exists", os.path.join(root.replace("json", "sentiment"), f"{file}"))
                    continue
                try:
                    df_part = pd.read_json(os.path.join(root, file))
                except:
                    df_part = pd.read_json(os.path.join(root, file), lines=True)
                #df_part = df_part[["body", "id", "subreddit"]]
                try:
                    df_part["group"] = df_part["subreddit"].str.lower().map(subreddits.subreddit_to_group)
                except:
                    print(df_part)
                    exit()
                df_part = df_part.dropna(subset=["group"])
                df_part = df_part[df_part["group"].isin(["Democrat", "Republican"])]
                df_part["sentiment_score"] = df_part.apply(lambda row: sia.polarity_scores(row["body"])["compound"], axis=1)
                df_part.to_json(os.path.join(root.replace("json", "sentiment"), f"{file}"))

if __name__ == '__main__':
    overall_sentiment()

