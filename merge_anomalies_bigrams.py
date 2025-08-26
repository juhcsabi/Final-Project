import collections
import json
import os
import pprint

import numpy
import pandas as pd

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
anomaly_folders = ["tss_toxicities", "tss_community_feedback", "time_series", "time_series_jaccard", "time_series_overlap_coef"]
anomaly_files = ["attention_flow_anomalies.csv", "sentiment_evolutions_anomalies.csv"]

def merge_anomalies():
    group = "Democrat"
    df = pd.read_csv(os.path.join(data_path, f"anomalic_bigrams_on_events_{group}.csv"), index_col=0, parse_dates=True)
    df = df.loc[:, df.any()]
    anomaly_dates = collections.defaultdict(dict)
    dates = []
    for folder in anomaly_folders:
        for root, dirs, files in os.walk(os.path.join(data_path, folder)):
            for file in files:
                if file.endswith(".csv") and group in file:
                    filename = os.path.splitext(file)[0]
                    anomalies = pd.read_csv(os.path.join(root, file), index_col=0, parse_dates=True)["anomaly"]
                    for bigram in df.columns:
                        df_and = df[bigram] & anomalies
                        if len(anomalies[anomalies]) > 0:
                            dates.extend(list(anomalies[anomalies].index.values))
                        if len(df_and[df_and]) > 0:
                            if filename not in anomaly_dates[folder]:
                                anomaly_dates[folder][filename] = collections.defaultdict(dict)
                            anomaly_dates[folder][filename][bigram] = list([(date.year, date.month) for date in df_and[df_and].index])
                            #pprint.pprint(anomaly_dates, indent=4)
                elif file.endswith(".csv"):
                    print(file)
    for file in anomaly_files:
        if file.endswith(".csv"):
            filename = os.path.splitext(file)[0]
            anomalies = pd.read_csv(os.path.join(data_path, file), index_col=0, parse_dates=True)
            for bigram in df.columns:
                for col in anomalies.columns:
                    if group in col:
                        df_and = df[bigram] & anomalies[col]
                        if len(anomalies[col][anomalies[col]]) > 0:
                            dates.extend(list(anomalies[anomalies].index.values))
                        if len(df_and[df_and]) > 0:
                            if filename not in anomaly_dates[os.path.splitext(file)[0]]:
                                anomaly_dates[os.path.splitext(file)[0]][col] = collections.defaultdict(dict)
                            anomaly_dates[os.path.splitext(file)[0]][col][bigram] = list(
                                [(date.year, date.month) for date in df_and[df_and].index])
                    else:
                        print(col)

    #pprint.pprint(anomaly_dates, indent=4)
    with open(os.path.join(data_path, f"anomalic_events_bigrams_behaviors_merged_{group}.json"), "wt") as f:
        json.dump(anomaly_dates, f)

def count_anomalies_by_month():
    group = "Democrat"
    #df = pd.read_csv(os.path.join(data_path, f"anomalic_bigrams_on_events_{group}.csv"), index_col=0, parse_dates=True)
    df = pd.DataFrame()
    anomaly_dates = collections.defaultdict(dict)
    dates = []
    for folder in anomaly_folders:
        for root, dirs, files in os.walk(os.path.join(data_path, folder)):
            for file in files:
                if file.endswith(".csv") and group in file:
                    filename = os.path.splitext(file)[0]
                    anomalies = pd.read_csv(os.path.join(root, file), index_col=0, parse_dates=True)["anomaly"]
                    df[os.path.join(root, file)] = anomalies
                elif file.endswith(".csv"):
                    print(file)
    for file in anomaly_files:
        if file.endswith(".csv"):
            filename = os.path.splitext(file)[0]
            anomalies = pd.read_csv(os.path.join(data_path, file), index_col=0, parse_dates=True)
            for bigram in df.columns:
                for col in anomalies.columns:
                    if group in col:
                        df[f"{filename}-{col}"] = anomalies[col]
                        print(anomalies[col])
                    else:
                        print(col)

    cnts = []
    for idx, row in df.iterrows():
        print(row)
        cnts.append(len(row[row.fillna(False)]))
    print(cnts)

    df_cnts = pd.DataFrame(index=df.index)
    df_cnts["cnts"] = cnts

    df_cnts.to_csv(os.path.join(data_path, f"anomaly_counts_{group}.csv"))


count_anomalies_by_month()

