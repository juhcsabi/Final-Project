import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
import ruptures as rpt
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt

import subreddits
from sentiment import load_custom_lexicon

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
json_path = os.path.join(data_path, "json")


def load_json_files(folders, nr_of_partitions=8):
    dfs = [pd.DataFrame() for _ in range(nr_of_partitions)]
    for folder in folders:
        for root, _, files in os.walk(os.path.join(json_path, folder)):
            for i, file in enumerate(files):
                if not file.endswith(".json"):
                    continue
                partition = i % nr_of_partitions
                df = pd.read_json(os.path.join(root, file), lines=True)
                dfs[partition] = pd.concat([dfs[partition], df])
    return pd.concat(dfs)


def process_dataframe(df):
    df['converted_date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
    df["group"] = df["subreddit"].str.lower().map(subreddits.subreddit_to_group)
    df.drop(["created_utc", "subreddit"], axis="columns", inplace=True)
    return df.groupby("converted_date")


def save_daily_csv(df_days):
    for key, group in df_days:
        group.to_csv(os.path.join(data_path, "json_daily", f"{key}.csv"))


def get_daily_granularity():
    folders = ["2013-01"]
    df = load_json_files(folders)
    df_days = process_dataframe(df)
    save_daily_csv(df_days)


def load_graph_data(file_path):
    if os.path.exists(file_path):
        values = pd.read_csv(file_path)["value"].values
        return values[len(values) // 2:], values[:len(values) // 2]
    return None, None


def compute_edge_weight(community1_authors, community2_authors, method):
    intersection = len(community1_authors.intersection(community2_authors))
    if method == "normal":
        return intersection / len(community1_authors)
    elif method == "jaccard":
        return intersection / len(community1_authors.union(community2_authors))
    return intersection / min(len(community1_authors), len(community2_authors))


def process_graph_files(root, files, chosen_date, community1, community2, method):
    pre, post = [], []
    pre_users_comm1, pre_users_comm2 = set(), set()
    post_users_comm1, post_users_comm2 = set(), set()
    dates, date_objs = [], []

    for file in files:
        date_str = os.path.splitext(file)[0]
        date_object = datetime.strptime(date_str, "%Y-%m-%d").date()
        difference = chosen_date - date_object

        if not (-27 <= difference.days <= 28):
            continue

        before = difference.days > 0
        dates.append(date_str)
        date_objs.append(date_object)

        df = pd.read_csv(os.path.join(root, file))
        df_groups = df.groupby("group")

        community1_authors = set(df_groups.get_group(community1)["author"].unique())
        community2_authors = set(df_groups.get_group(community2)["author"].unique())
        edge_weight = compute_edge_weight(community1_authors, community2_authors, method)

        if before:
            pre.append(edge_weight)
            pre_users_comm1.update(community1_authors)
            pre_users_comm2.update(community2_authors)
        else:
            post.append(edge_weight)
            post_users_comm1.update(community1_authors)
            post_users_comm2.update(community2_authors)

    return pre, post, dates, date_objs


def generate_plot(dates, values, title, y_title, method, community1, community2, chosen_date):
    trace = go.Scatter(x=dates, y=values, mode='lines+markers', name='Values')
    layout = go.Layout(title=title, xaxis=dict(title='Date'), yaxis=dict(title=y_title))
    fig = go.Figure(data=[trace], layout=layout)

    base_filename = f"{community1} - {community2} - {method if method else 'basic'} - {chosen_date}"
    fig.write_html(os.path.join(data_path, "event_daily_plots", f"{base_filename}.html"))
    fig.write_image(os.path.join(data_path, "event_daily_plots", f"{base_filename}.png"))


def run_statistical_tests(pre, post):
    t_stat, p_value = ttest_ind(pre, post, equal_var=False)
    print(f"t-Test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    u_stat, p_value_u = mannwhitneyu(pre, post)
    print(f"Mann-Whitney U Test: U-statistic = {u_stat:.4f}, p-value = {p_value_u:.4f}")

    print(shapiro(pre))
    print(shapiro(post))


def detect_change_points(values):
    algo = rpt.Binseg(model="l2").fit(np.array(values))
    result = algo.predict(n_bkps=2)
    print("Change Point Detection:", result)


def save_graph_results(df_values, method, community1, community2, chosen_date, overwrite):
    os.makedirs(os.path.join(data_path, "results", "graphs", method), exist_ok=True)
    file_path = os.path.join(data_path, "results", "graphs", method, f"{community1} - {community2} - {chosen_date}.csv")
    if not os.path.exists(file_path) or overwrite:
        df_values.to_csv(file_path)


def check_graph_for_date(year, month, day, community1, community2, method="", overwrite=False):
    chosen_date = date(year, month, day)
    method = method or "normal"
    file_path = os.path.join(data_path, "results", "graphs", method, f"{community1} - {community2} - {chosen_date}.csv")

    pre, post = load_graph_data(file_path)

    if pre is None and post is None:
        for root, _, files in os.walk(os.path.join(data_path, "json_daily")):
            pre, post, dates, date_objs = process_graph_files(root, files, chosen_date, community1, community2, method)
            break

        values = pre + post
        generate_plot(dates, values, f'{community1} - {community2} - {method} - {chosen_date}', 'Value', method,
                      community1, community2, chosen_date)

        run_statistical_tests(pre, post)
        detect_change_points(values)

        df_values = pd.DataFrame(index=pd.DatetimeIndex(date_objs))
        df_values["value"] = values
        save_graph_results(df_values, method, community1, community2, chosen_date, overwrite)
    else:
        print("Graph data already exists, skipping computation.")


def load_sentiment_data(file_path):
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
        values = df_existing["value"].values
        return values[len(values) // 2:], values[:len(values) // 2], df_existing.index.values
    return None, None, []


def calculate_sentiment_scores(df, sia, method):
    df["sentiment_score"] = df["body"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

    if method == "mean":
        return df["sentiment_score"].mean()
    elif method == "median":
        return df["sentiment_score"].median()
    return df["sentiment_score"].std()


def process_sentiment_files(root, files, chosen_date, community1, sia, method):
    pre, post, dates, date_objs = [], [], [], []

    for file in files:
        date_str = os.path.splitext(file)[0]
        date_object = datetime.strptime(date_str, "%Y-%m-%d").date()
        difference = chosen_date - date_object

        if not (-27 <= difference.days <= 28):
            continue

        before = difference.days > 0
        dates.append(date_str)
        date_objs.append(date_object)

        df = pd.read_csv(os.path.join(root, file))
        df_groups = df.groupby("group")
        df_chosen_group = df_groups.get_group(community1).copy(deep=True)
        daily_value = calculate_sentiment_scores(df_chosen_group, sia, method)

        if before:
            pre.append(daily_value)
        else:
            post.append(daily_value)

    return pre, post, dates, date_objs


def save_sentiment_results(df_values, method, community1, chosen_date, overwrite):
    os.makedirs(os.path.join(data_path, "results", "sentiments", method), exist_ok=True)
    file_path = os.path.join(data_path, "results", "sentiments", method, f"{community1} - {chosen_date}.csv")
    if not os.path.exists(file_path) or overwrite:
        df_values.to_csv(file_path)


def check_sentiment_for_date(year, month, day, community1, method="mean", overwrite=False):
    chosen_date = date(year, month, day)
    file_path = os.path.join(data_path, "results", "sentiments", method, f"{community1} - {chosen_date}.csv")

    pre, post, df_index = load_sentiment_data(file_path)

    if pre is None and post is None:
        sia = load_custom_lexicon()
        for root, _, files in os.walk(os.path.join(data_path, "json_daily")):
            pre, post, dates, date_objs = process_sentiment_files(root, files, chosen_date, community1, sia, method)
            break

        values = pre + post
        generate_plot(dates, values, f'{community1} - {method} - {chosen_date}', 'Sentiment Score', method, community1,
                      "", chosen_date)

        run_statistical_tests(pre, post)
        detect_change_points(values)

        df_values = pd.DataFrame(index=pd.DatetimeIndex(date_objs))
        df_values["value"] = values
        save_sentiment_results(df_values, method, community1, chosen_date, overwrite)
    else:
        print("Sentiment data already exists, skipping computation.")


check_sentiment_for_date(2012, 12, 14, "Republican", "mean")
check_sentiment_for_date(2012, 12, 14, "Democrat", "std")
check_graph_for_date(2017, 10, 1, "Republican", "Conspiracy", "")
check_graph_for_date(2014, 8, 9, "Democrat", "ChangeMyView", "")
check_graph_for_date(2020, 5, 25, "General_Science", "Democrat", "")
check_graph_for_date(2017, 10, 1, "Republican", "Conspiracy", "jaccard")
check_graph_for_date(2017, 4, 4, "Democrat", "Technology_Applied_Sciences", "jaccard")
check_feedback_for_date(2019, 8, 3, "Republican")
check_feedback_for_date(2019, 8, 3, "Democrat")