import collections
import json
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.util import bigrams
import string
from collections import Counter
import ast
import plotly.express as px

#from daily import check_graph_for_date

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
#nltk.download('stopwords')
#nltk.download('punkt')

def preprocess_and_generate_bigrams(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    bigram_list = list(bigrams(filtered_words))
    bigrams_to_delete = []
    for i in range(len(bigram_list)):
        if bigram_list[i][0] == "'s" and i > 0:
            bigram_list[i] = (bigram_list[i-1][0] + "'s", bigram_list[i][1])
            bigrams_to_delete.append(i - 1)
    indices = sorted(bigrams_to_delete, reverse=True)
    for index in indices:
        if 0 <= index < len(bigram_list):
            bigram_list.pop(index)
    return bigram_list


def event_bigrams():
    event_years = collections.defaultdict(dict)
    with open(os.path.join(data_path, "events.json"), "rt") as f:
        events = json.load(f)

    cnt = 0
    events_bigrammed = collections.defaultdict(dict)
    for month in events:
        for day in events[month]:
            events_bigrammed[month][day] = []
            event_years[month][day] = []
            for event in events[month][day]:
                (text, year) = event
                bigrams_list = preprocess_and_generate_bigrams(text)
                events_bigrammed[month][day].append(bigrams_list)
                event_years[month][day].append(year)
                cnt += len(bigrams_list)
    print(cnt)
    with open(os.path.join(data_path, "events_bigrammed.json"), "wt") as f:
        json.dump(events_bigrammed, f)
    with open(os.path.join(data_path, "events_bigrammed_years.json"), "wt") as f:
        json.dump(event_years, f)


def death_bigrams():
    cnt = 0
    death_years = collections.defaultdict(dict)
    with open(os.path.join(data_path, "deaths.json"), "rt") as f:
        deaths = json.load(f)
    deaths_bigrammed = collections.defaultdict(dict)
    for month in deaths:
        for day in deaths[month]:
            deaths_bigrammed[month][day] = []
            death_years[month][day] = []
            for death in deaths[month][day]:
                (text, year) = death
                bigrams_list = preprocess_and_generate_bigrams(text)
                deaths_bigrammed[month][day].append(bigrams_list)
                death_years[month][day].append(year)
                cnt += len(bigrams_list)
    print(cnt)
    with open(os.path.join(data_path, "deaths_bigrammed.json"), "wt") as f:
        json.dump(deaths_bigrammed, f)
    with open(os.path.join(data_path, "deaths_bigrammed_years.json"), "wt") as f:
        json.dump(death_years, f)



def count_bigrams_reddit(year, month):
    if os.path.exists(os.path.join(data_path, "bigrams_reddit", f"{year}-{month}.csv")):
        df = pd.read_csv(os.path.join(data_path, "bigrams_reddit", f"{year}-{month}.csv"))
        df = df[~df.isnull().any(axis=1)]
        df["bigram"] = df["bigram"].str.lower().apply(ast.literal_eval)
        df = df[~df['bigram'].apply(lambda x: "n't" in x)]
        df = df[df["count"] != 1]
        df = df[["count", "bigram"]]
        df.to_csv(os.path.join(data_path, "bigrams_reddit", f"{year}-{month}.csv"))
    else:
        df = pd.read_parquet(os.path.join(data_path, "json_filtered_active", f"{year}-{month}.parquet"))
        bigram_counter = Counter()

        i = 0
        for text in df['body']:
            if i % 1000 == 0:
                print(i / len(df))
            i += 1
            bigrams_list = preprocess_and_generate_bigrams(text)
            bigram_counter.update(bigrams_list)

        bigram_df = pd.DataFrame(bigram_counter.items(), columns=['bigram', 'count'])
        bigram_df = bigram_df[~bigram_df['bigram'].apply(lambda x: "n't" in x)]
        bigram_df = bigram_df[bigram_df["count"] != 1]

        bigram_df = bigram_df.sort_values(by='count', ascending=False).reset_index(drop=True)
        bigram_df.to_csv(os.path.join(data_path, "bigrams_reddit", f"{year}-{month}.csv"), index=False)
        print(bigram_df)

def all_reddit_bigrams():
    for year in range(2017, 2016, -1):
        for month in range(12, 0, -1):
            print(year, month)
            count_bigrams_reddit(str(year), str(month).zfill(2))


def merge_all_reddit_bigrams():
    for year in range(2021, 2011, -1):
        for month in range(12, 0, -1):
            print(year, month)
            merge_bigrams(str(year), str(month).zfill(2))


def merge_bigrams(year_file, month_file):

    bigram_directory = os.path.join(data_path, "bigrams_reddit")
    merged_bigram_directory = os.path.join(data_path, "merged_bigrams")
    if os.path.exists(os.path.join(merged_bigram_directory, f"{year_file}-{month_file}.csv")):
        return
    os.makedirs(merged_bigram_directory, exist_ok=True)
    rows = []
    with open(os.path.join(data_path, "events_bigrammed.json"), "rt") as f:
        data = json.load(f)
    for month, days in data.items():
        for day, tuples in days.items():
            for year_bigrams in tuples:
                for value1, value2 in year_bigrams:
                    rows.append({
                        'bigram': (value1[0].lower(), value1[1].lower()),
                    })

    df = pd.DataFrame(rows)

    df_reddit_bigrams = pd.read_csv(os.path.join(bigram_directory, f"{year_file}-{month_file}.csv"))
    df_reddit_bigrams = df_reddit_bigrams[["count", "bigram"]]
    df_reddit_bigrams = df_reddit_bigrams[~df_reddit_bigrams.isnull().any(axis=1)]
    df_reddit_bigrams["bigram"] = df_reddit_bigrams["bigram"].str.lower().apply(ast.literal_eval)

    df[['b1', 'b2']] = pd.DataFrame(df['bigram'].tolist(), index=df.index)
    df = df.drop_duplicates(subset=["b1", "b2"])

    df = df.drop("bigram", axis="columns")

    df_reddit_bigrams[['b1', 'b2']] = pd.DataFrame(df_reddit_bigrams['bigram'].tolist(), index=df_reddit_bigrams.index)
    df_reddit_bigrams = df_reddit_bigrams[["count", "b1", "b2"]]
    df_reddit_bigrams = df_reddit_bigrams.groupby(["b1", "b2"], as_index=False).sum("count")
    df = df.merge(df_reddit_bigrams, on=["b1", "b2"], how="left")
    df['count'].fillna(0, inplace=True)
    df['count'] = df['count'].astype(int)
    df = df.sort_values(by="count", ascending=False)
    df.to_csv(os.path.join(merged_bigram_directory, f"{year_file}-{month_file}.csv"))

def count_daily_bigrams_reddit(year, month):
    print(os.path.exists(os.path.join(data_path, "bigram_id_counts", f"{year}-{month}.csv")))
    print(os.path.join(data_path, "bigram_id_counts", f"{year}-{month}.csv"))
    if os.path.exists(os.path.join(data_path, "bigram_id_counts", f"{year}-{month}.csv")):
        df = pd.read_csv(os.path.join(data_path, "bigram_id_counts", f"{year}-{month}.csv"))
        df["bigram"] = df["bigram"].apply(ast.literal_eval)
        return df
    df = pd.read_parquet(os.path.join(data_path, "json_filtered_active", f"{year}-{month}.parquet"))
    bigram_counter = Counter()
    ids = collections.defaultdict(dict)
    with open(os.path.join(data_path, "deaths_bigrammed.json"), "rt") as f:
        deaths = json.load(f)
    with open(os.path.join(data_path, "events_bigrammed.json"), "rt") as f:
        events = json.load(f)

    relevant_bigrams = set()
    for _, day_dict in deaths.items():
        for day, bigrams in day_dict.items():
            bigrams = [(bigram[0][0], bigram[0][1]) for bigram in bigrams if len(bigram) > 0]
            relevant_bigrams.update(set(bigrams))
    for _, day_dict in events.items():
        for day, bigrams in day_dict.items():
            for year_bigrams in bigrams:
                tmp_bigrams = [(bigram[0], bigram[1]) for bigram in year_bigrams]
                relevant_bigrams.update(set(tmp_bigrams))


    i = 0
    for text in df['body']:
        if i % 1000 == 0:
            print(i / len(df))

        bigrams_list = preprocess_and_generate_bigrams(text)
        bigrams_set = [(bigram[0], bigram[1]) for bigram in bigrams_list]

        bigrams_to_keep = relevant_bigrams.intersection(set(bigrams_set))
        bigram_counter.update(bigrams_to_keep)
        try:
            id = df.iloc[i]["id"]
        except:
            print(i, len(df))
            exit()
        for bigram in bigrams_to_keep:
            if bigram not in ids[id]:
                ids[id][bigram] = 1
            else:
                ids[id][bigram] += 1
        i += 1

    bigram_df = pd.DataFrame(bigram_counter.items(), columns=['bigram', 'count'])
    bigram_df = bigram_df[~bigram_df['bigram'].apply(lambda x: "n't" in x)]
    #bigram_df = bigram_df[bigram_df["count"] != 1]

    bigram_df = bigram_df.sort_values(by='count', ascending=False).reset_index(drop=True)
    bigram_df.to_csv(os.path.join(data_path, "bigrams_reddit_relevant", f"{year}-{month}.csv"), index=False)
    rows = []
    for id, bigram_dict in ids.items():
        for bigram, count in bigram_dict.items():
            rows.append({
                'bigram': (bigram[0].lower(), bigram[1].lower()),
                'id': id,
                'count': count
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_path, "bigram_id_counts", f"{year}-{month}.csv"))
    return df


def create_all_daily_bigrams():
    #event_bigrams()
    #death_bigrams()
    for year in range(2022, 2023, 1):
        for month in range(1, 13, 1):
            print(year, month)
            id_counts_df = count_daily_bigrams_reddit(str(year), str(month).zfill(2))
            for root, dirs, files in os.walk(os.path.join(data_path, "json", f"{year}-{str(month).zfill(2)}")):
                monthly_df = pd.DataFrame()
                for file in files:
                    print(file)
                    if ".json" not in file:
                        continue
                    try:
                        df = pd.read_json(os.path.join(root, file))
                    except:
                        df = pd.read_json(os.path.join(root, file), lines=True)
                    df = df.merge(id_counts_df, on="id", how="inner")
                    df = df[["id", "created_utc", "subreddit", "bigram", "count"]]
                    df = df[df["bigram"] != ("''", '``')]
                    df['date'] = pd.to_datetime(df['created_utc'], unit="s", utc=True).dt.date
                    if len(monthly_df) == 0:
                        monthly_df = df
                    else:
                        monthly_df = pd.concat([monthly_df, df])
                monthly_df.to_csv(os.path.join(data_path, "timestamped_bigrams", f"{year}-{str(month).zfill(2)}.csv"))
                monthly_df.groupby(["date", "bigram"])["count"].sum().reset_index().to_csv(os.path.join(data_path, "daily_bigrams", f"{year}-{str(month).zfill(2)}.csv"))
                #exit()

def visualize():
    df = pd.DataFrame()

    df_comm_fb = pd.read_csv(os.path.join(data_path, "community_feedback.csv"))

    comm_counts = df_comm_fb[df_comm_fb["Group"] == "Democrat"]["Number of comments"].values + \
                  df_comm_fb[df_comm_fb["Group"] == "Republican"]["Number of comments"].values
    for root, dirs, files in os.walk(os.path.join(data_path, "bigrams_reddit_relevant")):
        for i, file in enumerate(files):
            print(file)
            df_month = pd.read_csv(os.path.join(root, file))
            df_month["count"] = df_month["count"] / comm_counts[i]
            df_month.rename(columns={"count": os.path.splitext(file)[0]}, inplace=True)
            if len(df) == 0:
                df = df_month
            else:
                df = df.merge(df_month, how="outer", on="bigram")
    print(df)
    df.to_csv(os.path.join(data_path, "bigrams_reddit_relevant_all.csv"))

    exit()

    filtered_df = df[df.iloc[:, 1:].max(axis=1) > 0.001]

    df_long = filtered_df.melt(id_vars=['bigram'], var_name='Date', value_name='Value')

    df_long['Date'] = pd.to_datetime(df_long['Date'])

    df_long = df_long.fillna(0)

    fig = px.line(df_long, x='Date', y='Value', color='bigram', title='Time Series of Values')
    fig.show()


def threshold_bigrams(filename):
    df = pd.read_csv(os.path.join(data_path, filename), index_col=0).fillna(0)

    filtered_df = df[df.iloc[:, 1:].max(axis=1) > 0.001]

    filtered_df.set_index("bigram", inplace=True)

    filtered_df = filtered_df.transpose()

    print(filtered_df.index)

    filtered_df.index = pd.to_datetime(filtered_df.index, format='%Y-%m')

    print(filtered_df)

    filtered_df.to_csv(os.path.join(data_path, f"{filename}_thresholded.csv"))


def get_relevant_bigram_months(group):
    with open(os.path.join(data_path, "events_bigrammed.json"), "rt") as f:
        events = json.load(f)
    with open(os.path.join(data_path, "events_bigrammed_years.json"), "rt") as f:
        event_years = json.load(f)
    df_anomalies_existing = pd.read_csv(os.path.join(data_path, f"bigram_counts_{group}_thresholded_anomalies.csv"),
                                        index_col=0, parse_dates=True)
    for col in df_anomalies_existing.columns:
        df_anomalies_existing[col].values[:] = False
    columns = df_anomalies_existing.columns.values
    for month in events:
        all_bigrams_month = []
        all_bigrams_years = []
        for day in events[month]:
            for i, bigram_list in enumerate(events[month][day]):
                all_bigrams_month.extend(bigram_list)
                all_bigrams_years.extend([event_years[month][day][i] for _ in range(len(bigram_list))])
        for i in range(len(all_bigrams_month)):
            (token1, token2) = all_bigrams_month[i]
            all_bigrams_month[i] = str((token1, token2))
        relevant_bigrams_month_year = []
        for i, bigram in enumerate(all_bigrams_month):
            if bigram in columns:
                relevant_bigrams_month_year.append((bigram, all_bigrams_years[i]))
        #print(relevant_bigrams_month_year)
        for (bigram, year) in relevant_bigrams_month_year:
            print(df_anomalies_existing[bigram])
            for date in df_anomalies_existing.index:
                if int(date.year) == int(year) and int(date.month) == int(month):
                    df_anomalies_existing[bigram].at[date] = True
            #print(df_anomalies_existing[bigram].values)
        df_anomalies_existing.to_csv(os.path.join(data_path, f"bigram_dates_{group}.csv"))


def anomalic_bigrams_on_events(group):
    df_dates = pd.read_csv(os.path.join(data_path, f"bigram_dates_{group}.csv"), index_col=0, parse_dates=True)
    df_anomalic_bigrams = pd.read_csv(os.path.join(data_path, f"bigram_counts_{group}_thresholded_anomalies.csv"), index_col=0, parse_dates=True)
    df_dates_and_anomalic_bigrams = pd.DataFrame()
    for col in df_dates.columns:
        df_dates_and_anomalic_bigrams[col] = df_dates[col] & df_anomalic_bigrams[col]
    df_dates_and_anomalic_bigrams.to_csv(os.path.join(data_path, f"anomalic_bigrams_on_events_{group}.csv"))


def dunno():
    df = pd.read_csv(os.path.join(data_path, "bigrams_reddit_relevant_all.csv"), index_col=0, parse_dates=True)

    filtered_df = df[df.iloc[:, 1:].max(axis=1) > 0.001]

    df_long = filtered_df.melt(id_vars=['bigram'], var_name='Date', value_name='Value')

    # Convert the Date column to datetime
    df_long['Date'] = pd.to_datetime(df_long['Date'])

    df_long = df_long.fillna(0)

    # Create the line graph
    fig = px.line(df_long, x='Date', y='Value', color='bigram', title='Time Series of Values')
    fig.show()


#get_relevant_bigram_months()
#anomalic_bigrams_on_events()
#check_graph_for_date(2017, 10, 1, "Conspiracy", "Republican", "jaccard")
#event_bigrams()
#death_bigrams()

def create_bigram_counts_by_group():
    overall_bigram_counts = {}
    for year in range(2012, 2023, 1):
        for month in range(1, 13, 1):
            print(year, month)
            id_counts_df = count_daily_bigrams_reddit(str(year), str(month).zfill(2))
            df = pd.read_parquet(os.path.join(data_path, "json_filtered_active", f"{year}-{str(month).zfill(2)}.parquet"))
            joined_df = df.merge(id_counts_df, on="id", how="inner")
            joined_df = joined_df[["id", "group", "count", "bigram"]]
            community_bigrams = joined_df.groupby("group")
            for community_idx, community_df in community_bigrams:
                os.makedirs(os.path.join(data_path, f"bigram_counts_{community_idx}"), exist_ok=True)
                bigram_counts = community_df.groupby("bigram").sum()["count"].reset_index().sort_values(by="count", ascending=False)
                bigram_counts = bigram_counts[~bigram_counts["bigram"].astype(str).str.contains("''")]
                bigram_counts = bigram_counts[~bigram_counts["bigram"].astype(str).str.contains("n't")]
                bigram_counts["count"] = bigram_counts["count"] / len(df[df["group"] == community_idx])
                bigram_counts.to_csv(os.path.join(data_path, f"bigram_counts_{community_idx}", f"{year}-{str(month).zfill(2)}.csv"))
                bigram_counts = bigram_counts.rename({"count": f"{year}-{str(month).zfill(2)}"}, axis="columns")
                if community_idx not in overall_bigram_counts:
                    overall_bigram_counts[community_idx] = bigram_counts
                else:
                    overall_bigram_counts[community_idx] = overall_bigram_counts[community_idx].merge(bigram_counts, how="outer", on="bigram")
            for com, bigram_count in overall_bigram_counts.items():
                bigram_count.to_csv(os.path.join(data_path, f"bigram_counts_{com}.csv"))


if __name__ == '__main__':

    for group in ["Democrat", "Republican"]:
        get_relevant_bigram_months(group)
        anomalic_bigrams_on_events(group)