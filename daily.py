import math
import os
import pprint
from collections import Counter
from datetime import datetime, date

from numpy.linalg import norm
import statsmodels.api as sm

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, ttest_rel, wilcoxon, zscore
from scipy.linalg import toeplitz
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.multitest import multipletests

import subreddits
from bigrams import preprocess_and_generate_bigrams
from sentiment import load_custom_lexicon

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
json_path = os.path.join(data_path, "json2")
json_filtered_path = os.path.join(data_path, "json_filtered_active")
com_cnt = len(subreddits.subreddit_lists)
toxicities_path = os.path.join(data_path, "csv")
CMNTS_PER_1_PC = "nr_authors_1_pc"
NR_AUTHORS_1_PC = "cmnts_in_1_pc"
CMNT_PER_AUTHOR = "cmnt_per_author"
NR_CMNTS = "nr_cmnts"
NR_AUTHORS = "nr_authors"
RATIO_AUTHORS_RESP_SUM80= "ratio_authors_resp_sum80"


def get_daily_granularity(toxicity=False, toxicity_type="threat"):
    folders = ["2017-01", "2017-02", "2020-03"]
    for folder in folders:
        nr_of_partitions = 16
        dfs = [pd.DataFrame() for _ in range(nr_of_partitions)]
        df = pd.DataFrame()
        for root, dirs, files in os.walk(os.path.join(json_path, folder)):
            for i, file in enumerate(files):
                if ".json" not in file:
                    continue
                print(folder, file)
                df_part = pd.read_json(os.path.join(root, file), lines=True)
                if len(dfs[i % nr_of_partitions]) == 0:
                    dfs[i % nr_of_partitions] = df_part
                else:
                    dfs[i % nr_of_partitions] = pd.concat([dfs[i % nr_of_partitions], df_part])
            for i in range(nr_of_partitions):
                dfs[i]["group"] = dfs[i]["subreddit"].str.lower().map(subreddits.subreddit_to_group)
                dfs[i].drop(["subreddit"], axis="columns", inplace=True)
                if toxicity:
                    dfs[i] = dfs[i][dfs[i]["group"].isin(["Democrat", "Republican"])]
                if len(df) == 0:
                    df = dfs[i]
                else:
                    df = pd.concat([df, dfs[i]])
        print(df)
        df['converted_date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
        #df["group"] = df["subreddit"].str.lower().map(subreddits.subreddit_to_group)
        df.drop(["created_utc"], axis="columns", inplace=True)
        df.to_csv(os.path.join(data_path, f"{folder}.csv"))
        df_days = df.groupby("converted_date")
        for key, group in df_days:
            print(key)
            print(group)
            try:
                group.to_csv(os.path.join(data_path, "json_daily", f"{key}.csv"))
            except:
                print("faszkivan")


def join_toxicities():
    months = ["2022-03"]
    for pf in ["_r", "_d"]:
        for month in months:

            tox_df = pd.read_csv(os.path.join(toxicities_path, f"{month}{pf}.csv"), engine="python", on_bad_lines='error')
            tox_df = tox_df.drop(["body", "group", "index"], axis="columns")
            #for index, row in tox_df.iterrows():
            #    print(index, row)
            for root, dirs, files in os.walk(os.path.join(data_path, "json_daily")):
                for file in files:
                    if not file.startswith(month):
                        continue
                    daily_df = pd.read_csv(os.path.join(root, file), index_col=0)
                    merged_df = tox_df.merge(daily_df, on="id")
                    print(merged_df)
                    merged_df.to_csv(os.path.join(root, f"tox_{file}"))


def load_existing_behavioral_values(df_path):
    df_existing = pd.read_csv(
        df_path, index_col=0,
        parse_dates=True)
    values = df_existing["value"].values
    dates = df_existing.index.values
    pre = values[:len(values)//2]
    post = values[len(values)//2:]
    return pre, post, values, dates, dates


def get_daily_data(root, file, chosen_date, dates, date_objs, more_before=0, more_after=0, meta=False, group=None, new=True, time_period=28):
    date_str = os.path.splitext(file)[0].replace("tox_", "")
    date_object = datetime.strptime(date_str, "%Y-%m-%d").date()
    difference = chosen_date - date_object
    if 0 < difference.days <= time_period + more_before:
        before = True
    elif 0 >= difference.days > -time_period - more_after:
        before = False
    else:
        return dates, date_objs, None, None, None
    dates.append(date_str)
    date_objs.append(date_object)
    df = pd.read_csv(os.path.join(root, file))
    if meta:
        if not os.path.exists(os.path.join(data_path, "json_daily_meta", f"{group}_{'new' if new else 'old'}_{file}")):
            if not os.path.exists(os.path.join(data_path, "monthly_users_meta", f"{group}_old_{file}")):
                df_users = pd.read_csv(os.path.join(data_path, f"{group}_users.csv"))
                df_users = df_users[["author", "joined"]]
                df_users["year_joined"] = df_users["joined"].str.split("-").str[0].astype(int)
                df_users["month_joined"] = df_users["joined"].str.split("-").str[1].astype(int)
                if chosen_date.month < 12:
                    df_users = df_users[df_users["year_joined"] <= chosen_date.year]
                    df_bool = (df_users["year_joined"] == chosen_date.year) & ((df_users["month_joined"] - chosen_date.month).abs() <= 1)
                else:
                    df_users = df_users[(df_users["year_joined"] <= chosen_date.year) |
                                        ((df_users["year_joined"] == chosen_date.year + 1) & (df_users["month_joined"] == 1))]
                    df_bool = ((df_users["year_joined"] == chosen_date.year) & ((df_users["month_joined"] - chosen_date.month).abs() <= 1)) | \
                              (df_users["year_joined"] == chosen_date.year + 1)

                df_users_new = df_users[df_bool]
                df_users_old = df_users[~df_bool]
                df_users_old.to_csv(os.path.join(data_path, "monthly_users_meta", f"{group}_old_{chosen_date.year}_{chosen_date.month}.csv"))
                df_users_new.to_csv(os.path.join(data_path, "monthly_users_meta", f"{group}_new_{chosen_date.year}_{chosen_date.month}.csv"))
            else:
                df_users_old = pd.read_csv(os.path.join(data_path, "monthly_users_meta", f"{group}_old_{chosen_date.year}_{chosen_date.month}.csv"))
                df_users_new = pd.read_csv(os.path.join(data_path, "monthly_users_meta", f"{group}_new_{chosen_date.year}_{chosen_date.month}.csv"))
            df_new = df[(df["author"].isin(df_users_new["author"])) | (df["group"] != group)]
            df_old = df[(df["author"].isin(df_users_old["author"])) | (df["group"] != group)]
            df_new.to_csv(os.path.join(data_path, "json_daily_meta", f"{group}_new_{file}"))
            df_old.to_csv(os.path.join(data_path, "json_daily_meta", f"{group}_old_{file}"))
            df = df_new if new else df_old
        else:
            df = pd.read_csv(os.path.join(data_path, "json_daily_meta", f"{group}_{'new' if new else 'old'}_{file}"))
    # bigram_counter = Counter()
    # for text in df['body'].dropna():
    #     try:
    #         bigrams_list = preprocess_and_generate_bigrams(text)
    #     except:
    #         print(text)
    #         exit()
    #     bigram_counter.update(bigrams_list)
    df_groups = df.groupby("group")
    #print(bigram_counter)
    #print(dates, date_objs, df, df_groups, before)
    return dates, date_objs, df, df_groups, before#, bigram_counter[("sandy", "hook")]

def statistical_test(pre, post, dates, chosen_date, get_gls=True, get_gls_sigma=True, time_period=28):
    print_summary(pre, post)
    if get_gls:
        days = np.arange(-len(pre), len(post))
        #print(len(days), len(pre), len(post))
        data = pd.DataFrame({
            'day': days,
            'values': list(pre) + list(post),
            'post_event': [0 if d < 0 else 1 for d in days]
        })

        data['day_post_event'] = data['day'] * data['post_event']

        model = sm.GLS(data['values'], sm.add_constant(data[['day', 'post_event', 'day_post_event']]))
        result = model.fit()

        #print("R-squared", result.rsquared)
        #print("F-value", result.fvalue, "F probability", result.f_pvalue)
        #print(result.summary())
        if get_gls_sigma:
            y = np.r_[pre, post]
            n_pre = len(pre)
            days = np.arange(-n_pre, len(post))  # 0 = first post-event obs

            X = pd.DataFrame({
                "const": 1.0,
                "day": days,
                "post": (days >= 0).astype(int)
            })
            X["day_post"] = X["day"] * X["post"]
            dates = pd.DatetimeIndex(dates)
            dow = dates.dayofweek
            dummies = pd.get_dummies(dow, prefix='dow', drop_first=True)
            X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            X = X.astype(float)

            #print(X)
            #print(y)

            # ─── 1. OLS pass -- get an AR(1) estimate from residuals ────────────────
            ols = sm.OLS(y, X).fit()
            rho = sm.tsa.acf(ols.resid, nlags=1)[1]  # φ̂₁

            # ─── 2. Build Toeplitz Σ  (Σ_ij = ρ^{|i-j|})  and run GLS ──────────────
            T = len(y)
            Sigma = toeplitz(rho ** np.arange(T))
            #print(f"Rho: {rho}, T: {T}")
            #for row in Sigma:
            #    row_str = ""
            #    for value in row:
            #        row_str = f"{row_str},{round(value, 4)}"
            #    print(row_str)
            #print(X)
            #print(y)
            gls = sm.GLS(y, X, sigma=Sigma).fit()

            # ─── 3. Joint Wald/F test:  post = day_post = 0  ────────────────────────
            wald = gls.f_test("post = 0, day_post = 0")

            if True:
                #print(gls.summary())
                print(f"Wald F({int(wald.df_num)}, {int(wald.df_denom)}) = "
                      f"{float(wald.fvalue):.3f},  p = {float(wald.pvalue):.4g}")
                return wald.fvalue, wald.pvalue, result
        return result

    else:

        pre = np.array([element for element in reversed(list(pre))])
        z_scores = np.abs(zscore(pre-post))
        outliers = (pre-post)[z_scores > 3]
        print(outliers)
        #print(shapiro(pre - post).pvalue)
        if shapiro(pre - post).pvalue > 0.05:
            t_statistic, p_value_t = ttest_rel(post, pre)
            print(f"Paired t-Test: t-statistic = {t_statistic}, p-value = {p_value_t}")
            if p_value_t < 0.05:
                print("Means deviate")
            else:
                print("Means do not deviate")

        else:
            w_statistic, p_value_w = wilcoxon(post, pre)
            print(f"Wilcoxon Signed-Ranks Test: W-statistic = {w_statistic}, p-value = {p_value_w}")
            if p_value_w < 0.05:
                print("Means deviate")
            else:
                print("Means do not deviate")


def plot_interrupted_tss(dates, values, result, method, filename_no_ext):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='markers',
        name='Observed Values',
        marker=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=result.fittedvalues,
        mode='lines',
        name='Fitted Values',
        line=dict(color='orange')
    ))

    # Add a vertical line to mark the event
    fig.add_vline(x=dates[28], line=dict(color='red', dash='dash'), name='Event')

    # Customize the layout
    fig.update_layout(
        title='Interrupted Time Series Analysis',
        xaxis_title='Day',
        yaxis_title=method,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            showticklabels=True,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            showticklabels=True,
        ),
        template='plotly_white'
    )

    #fig.show()
    fig.write_html(f"{os.path.join(data_path, 'event_daily_plots', 'interrupted', 'html',  filename_no_ext)}.html")
    fig.write_image(f"{os.path.join(data_path, 'event_daily_plots', 'interrupted', 'png',  filename_no_ext)}.png")


def print_summary(pre, post):
    values = np.array(list(pre) + list(post))
    coefficients = np.polyfit(np.arange(len(values)), values, 1)  # Degree 1 for linear
    values_trend = coefficients[0]
    print("Values mean", values.mean())
    print("Values min", values.min())
    print("Values max", values.max())
    print("Values trend", values_trend)
    coefficients = np.polyfit(np.arange(len(pre)), pre, 1)  # Degree 1 for linear
    pre_trend = coefficients[0]
    if type(pre) is list:
        pre = np.array(pre)
        post = np.array(post)
    print("Pre mean", pre.mean())
    print("Pre min", pre.min())
    print("Pre max", pre.max())
    print("Pre trend", pre_trend)
    coefficients = np.polyfit(np.arange(len(post)), post, 1)  # Degree 1 for linear
    post_trend = coefficients[0]
    print("Post mean", post.mean())
    print("Pre min", post.min())
    print("Pre max", post.max())
    print("Post trend", post_trend)
    print("Diff", pre[-1] - post[0])
    print("Mean Diff", pre.mean() - post.mean())
    print("Diff pct", (pre.mean() - post.mean()) / pre.mean())


def save_df(values, date_objs, filename):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    df_values = pd.DataFrame(index=pd.DatetimeIndex(date_objs))
    df_values["value"] = values
    df_values.to_csv(filename)

def plotly_plot(title, filename_no_ext, dates, values):
    trace = go.Scatter(x=dates, y=values, mode='lines+markers', name='Values')

    layout = go.Layout(
        title=title,
        xaxis=dict(title='Date'),
        yaxis=dict(title='Value'),
    )

    fig = go.Figure(data=[trace], layout=layout)

    # fig.show()
    fig.write_html(os.path.join(data_path, "event_daily_plots",
                                f"{filename_no_ext}.html"))
    fig.write_image(os.path.join(data_path, "event_daily_plots",
                                 f"{filename_no_ext}.png"))

def calculate_daily_graph_value(df_groups, community1, community2, method):
    community1_authors = set(df_groups.get_group(community1)["author"].unique())
    community2_authors = set(df_groups.get_group(community2)["author"].unique())
    if method == "normal":
        edge_weight = len(community1_authors.intersection(community2_authors)) / len(community1_authors)
    elif method == "jaccard":
        edge_weight = len(community1_authors.intersection(community2_authors)) / len(
            community1_authors.union(community2_authors))
    elif method=="overlap":
        edge_weight = len(community1_authors.intersection(community2_authors)) / min(len(community1_authors),
                                                                                     len(community2_authors))
    return edge_weight, community1_authors, community2_authors


def check_graph_for_date(year, month, day, community1, community2, method="", overwrite=False, meta=False, new=False):
    print(year, month, day, community1, community2, method, overwrite)
    chosen_date = date(year, month, day)
    method = "normal" if method == "" else method
    if os.path.exists(os.path.join(data_path, "results" if not meta else "meta", "graphs", method,
                         f"{community1} - {community2} - {chosen_date}.csv" if not meta else f"{'old' if not new else 'new'}_{community1} - {community2} - {chosen_date}.csv")) and not overwrite:
        pre, post, values, dates, date_objs = load_existing_behavioral_values(
            os.path.join(data_path, "results" if not meta else "meta", "graphs", method,
                         f"{community1} - {community2} - {chosen_date}.csv" if not meta else f"{'old' if not new else 'new'}_{community1} - {community2} - {chosen_date}.csv"))
    else:
        values = None
    if values is None:
        dates = []
        date_objs = []
        for root, dirs, files in os.walk(os.path.join(data_path, "json_daily")):
            pre = []
            pre_users_comm1 = set()
            pre_users_comm2 = set()
            post = []
            post_users_comm1 = set()
            post_users_comm2 = set()
            for file in files:
                if not meta:
                    dates, date_objs, df, df_groups, before = get_daily_data(root, file, chosen_date, dates, date_objs, meta=meta, new=new, group=community1)
                else:
                    chosen_community = community1 if community1 in ["Democrat", "Republican"] else community2
                    dates, date_objs, df, df_groups, before = get_daily_data(root, file, chosen_date, dates, date_objs, meta=meta, new=new, group=chosen_community)
                if df is None:
                    continue
                daily_value, community1_authors, community2_authors = calculate_daily_graph_value(df_groups, community1, community2, method)
                if before:
                    pre.append(daily_value)
                    pre_users_comm1 = pre_users_comm1.union(community1_authors)
                    pre_users_comm2 = pre_users_comm2.union(community2_authors)
                else:
                    post.append(daily_value)
                    post_users_comm1 = post_users_comm1.union(community1_authors)
                    post_users_comm2 = post_users_comm2.union(community2_authors)
        print_summary(pre, post)

        values = pre + post
    plotly_plot(f'{community1} - {community2} - {method} - {chosen_date}', os.path.join(data_path, "event_daily_plots", f"{community1} - {community2} - {method if method != '' else 'basic'} - {chosen_date}"), dates, values)
    fvalue, pvalue, res = statistical_test(pre, post, dates, chosen_date)
    plot_interrupted_tss(dates, values, res, f"Graph_{method}", f"{community1} - {community2} - {method if method != '' else 'basic'} - {chosen_date} - interrupted")

    df_filename = os.path.join(data_path, "results" if not meta else "meta", "graphs", method, ("" if not meta else "new_" if new else "old_") + f"{community1} - {community2} - {chosen_date}.csv")
    if not os.path.exists(df_filename):
        save_df(values, date_objs, df_filename)
    return {
        "date": chosen_date,
        "community1": community1,
        "community2": community2,
        "type": "graph",
        "method": method,
        "f-value": fvalue,
        "p-value": pvalue,
        "values": values
    }


def calculate_daily_sentiment_value(df_groups, community1, sia, method):
    df_chosen_group = df_groups.get_group(community1).copy(deep=True)

    df_chosen_group["sentiment_score"] = df_chosen_group.apply(
        lambda row: sia.polarity_scores(str(row["body"]))["compound"], axis=1)

    if method == "mean":
        daily_value = df_chosen_group["sentiment_score"].mean()
    elif method == "median":
        daily_value = df_chosen_group["sentiment_score"].median()
    else:
        daily_value = df_chosen_group["sentiment_score"].std()
    return daily_value


def check_sent_for_date(year, month, day, community1, method="mean", overwrite=False, meta=False, new=False):
    print(year, month, day, community1, method, overwrite)
    chosen_date = date(year, month, day)
    if os.path.exists(os.path.join(data_path, "results" if not meta else "meta", "sentiments", method,
                                   f"{community1} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {chosen_date}.csv")) and not overwrite:
        pre, post, values, dates, date_objs = load_existing_behavioral_values(os.path.join(data_path, "results" if not meta else "meta", "sentiments", method,
                                   f"{community1} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {chosen_date}.csv"))
    else:
        values = None
    sia = load_custom_lexicon()

    date_objs = []
    if values is None:
        dates = []
        for root, dirs, files in os.walk(os.path.join(data_path, "json_daily")):
            pre = []
            post = []
            for file in files:
                dates, date_objs, df, df_groups, before = get_daily_data(root, file, chosen_date, dates, date_objs, meta=meta, group=community1, new=new, more_before=0)
                if df is None:
                    continue
                #print(cnt)
                #exit()
                daily_value = calculate_daily_sentiment_value(df_groups, community1, sia, method)
                if before:
                    pre.append(daily_value)
                else:
                    post.append(daily_value)
        print_summary(pre, post)
        values = pre + post

    plotly_plot(f'{community1} - sentiment - {method} - {chosen_date}', os.path.join(data_path, "event_daily_plots", f"{community1} - sentiment - {method if method != '' else 'basic'} - {chosen_date}"), dates, values)

    fvalue, pvalue, res = statistical_test(pre, post, dates, chosen_date)
    plot_interrupted_tss(dates, values, res, f"Sentiment_{method}", f"{community1} - sentiment - {method if method != '' else 'basic'} - {chosen_date} - interrupted")
    df_filename = os.path.join(data_path, "results" if not meta else "meta", "sentiments", method, ("" if not meta else "new_" if new else "old_") + f"{community1} - {chosen_date}.csv")
    if not os.path.exists(df_filename) or overwrite:
        save_df(values, date_objs,
                df_filename)
    return {
        "date": chosen_date,
        "community": community1,
        "type": "sentiment",
        "method": method,
        "f-value": fvalue,
        "p-value": pvalue,
        "values": values,
    }


def calculate_daily_feedback_value(df_groups, community1, method, meta, full_df):
    df_chosen_group = df_groups.get_group(community1).copy(deep=True)
    if method == "nr_authors_1_pc":
        df_chosen_group = df_chosen_group.sort_values(by="score", ascending=False)
        top_1_percent = df_chosen_group.head(max(1, int(len(df_chosen_group) * 0.01)))
        daily_value = len(top_1_percent) / len(top_1_percent["author"].unique())
    elif method=="cmnt_per_author":
        nr_authors = len(df_chosen_group["author"].unique())
        daily_value = len(df_chosen_group) / nr_authors
    elif method == "nr_cmnts":
        daily_value = len(df_chosen_group)
    elif method == "nr_authors":
        daily_value = len(df_chosen_group["author"].unique())
    elif method == "ratio_authors_resp_sum80":
        df_chosen_group = df_chosen_group.sort_values(by="score", ascending=False)
        sum_scores = df_chosen_group["score"].sum()
        cul_scores = 0
        authors = set()
        for idx, row in df_chosen_group.iterrows():
            cul_scores += row["score"]
            authors.add(row["author"])
            if cul_scores >= 0.8 * sum_scores:
                break
        daily_value = len(authors) / len(df_chosen_group["author"].unique())
    else:
        if not meta or True:
            df_chosen_group = df_chosen_group.sort_values(by="score", ascending=False)
            top_1_percent = df_chosen_group.head(max(1, int(len(df_chosen_group) * 0.01)))
            daily_value = len(top_1_percent["author"].unique())
        else:
            full_df_groups = full_df.groupby("group")
            full_df_chosen_group = full_df_groups.get_group(community1).copy(deep=True)
            top_1_percent = full_df_chosen_group.head(max(1, int(len(df_chosen_group) * 0.01)))
            from_subusers_in_top_1_percent = top_1_percent[top_1_percent["author"].isin(df_chosen_group["author"])]
            try:
                daily_value = len(from_subusers_in_top_1_percent["author"].unique())/len(from_subusers_in_top_1_percent)
            except:
                daily_value = 0

    return daily_value


def check_feedback_for_date(year, month, day, community1, method="nr_authors_1_pc", overwrite=False, meta=False, new=False):
    print(year, month, day, community1, method, overwrite, meta)
    chosen_date = date(year, month, day)
    #print(os.path.join(data_path, "results" if not meta else "meta", "feedback", method,
    #                               f"{community1} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {chosen_date}.csv"))
    if os.path.exists(os.path.join(data_path, "results" if not meta else "meta", "feedback", method,
                                   f"{community1} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {chosen_date}.csv")) and not overwrite:
        pre, post, values, dates, date_objs = load_existing_behavioral_values(
            os.path.join(data_path, "results" if not meta else "meta", "feedback", method,
                         f"{community1} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {chosen_date}.csv"))
    else:
        values = None
    if values is None:
        date_objs = []
        dates = []
        for root, dirs, files in os.walk(os.path.join(data_path, "json_daily")):
            pre = []
            post = []
            for file in files:
                dates, date_objs, df, df_groups, before = get_daily_data(root, file, chosen_date, dates, date_objs, meta=meta, new=new, group=community1)

                if df is None:
                    continue
                daily_value = calculate_daily_feedback_value(df_groups, community1, method, meta=meta, full_df=pd.read_csv(os.path.join(root, file)))
                if before:
                    pre.append(daily_value)
                else:
                    post.append(daily_value)
        #print_summary(np.array(pre), np.array(post))
        values = pre + post

    plotly_plot(f'{community1} - sentiment - {method} - {chosen_date}', os.path.join(data_path, "event_daily_plots", f"{community1} - feedback - {method if method != '' else 'basic'} - {chosen_date}"), dates, values)
    fvalue, pvalue, res = statistical_test(pre, post, dates, chosen_date)
    if not meta:
        plot_interrupted_tss(dates, values, res, f"Feedback_{method}", f"{community1} - feedback - {method if method != '' else 'basic'} - {chosen_date} - interrupted")
    df_filename = os.path.join(data_path, "results" if not meta else "meta", "feedback", method, ("" if not meta else "new_" if new else "old_") + f"{community1} - {chosen_date}.csv")
    if not os.path.exists(df_filename) or overwrite:
        save_df(values, date_objs, os.path.join(data_path, "results" if not meta else "meta", "feedback", method, ("" if not meta else "new_" if new else "old_") + f"{community1} - {chosen_date}.csv"))
    return {
        "date": chosen_date,
        "community": community1,
        "type": "feedback",
        "method": method,
        "f-value": fvalue,
        "p-value": pvalue,
        "values": values
    }


def calculate_daily_attention_value(df_prior, df_after, community1, community2):
    users = set(df_prior["author"].values).intersection(set(df_after["author"].values))
    user_group_counts_prior = df_prior.groupby(["author", "group"]).size()
    C_prior = {user: [0 for _ in range(com_cnt)] for user in df_prior["author"].unique()}
    user_group_counts_after = df_after.groupby(["author", "group"]).size()
    C_after = {user: [0 for _ in range(com_cnt)] for user in df_after["author"].unique()}
    df_grouped_prior = user_group_counts_prior.reset_index(name='count')
    for g_index, group in df_grouped_prior.groupby("group"):
        ind = [key for key in subreddits.subreddit_lists.keys()].index(g_index)
        for r_index, row in group.iterrows():
            C_prior[row["author"]][ind] = row["count"]
    df_grouped_after = user_group_counts_after.reset_index(name='count')
    for g_index, group in df_grouped_after.groupby("group"):
        ind = [key for key in subreddits.subreddit_lists.keys()].index(g_index)
        for r_index, row in group.iterrows():
            C_after[row["author"]][ind] = row["count"]
    for key in C_prior:
        sum_prior = sum(C_prior[key])
        for k in range(len(C_prior[key])):
            C_prior[key][k] = C_prior[key][k] / sum_prior
    for key in C_after:
        sum_after = sum(C_after[key])
        for k in range(len(C_after[key])):
            C_after[key][k] = C_after[key][k] / sum_after
    b_matrix = {user: [0 for com in range(com_cnt)] for user in sorted(list(users))}
    for user in list(users):
        for j in range(com_cnt):
            if C_prior[user][j] == 0 or C_after[user][j] == 0:
                b_matrix[user][j] = C_prior[user][j] - C_after[user][j]
            else:
                b_matrix[user][j] = 0
    attention_flow = [[0 for _ in range(com_cnt)] for __ in range(com_cnt)]
    for user in b_matrix.keys():
        u_attention_flow = [[0 for _ in range(com_cnt)] for __ in range(com_cnt)]
        pos_flow = []
        pos_flow_indeces = []
        neg_flow = []
        neg_flow_indeces = []
        for i, b in enumerate(b_matrix[user]):
            if b > 0:
                pos_flow.append(b)
                pos_flow_indeces.append(i)
            elif b < 0:
                neg_flow.append(-b)
                neg_flow_indeces.append(i)
        neg_flow_normalized = norm(neg_flow, 1)
        for i in range(len(neg_flow)):
            neg_flow[i] = neg_flow[i] / neg_flow_normalized
        for pos_index, pos_flow_index in enumerate(pos_flow_indeces):
            for neg_index, neg_flow_index in enumerate(neg_flow_indeces):
                u_attention_flow[neg_flow_index][pos_flow_index] = neg_flow[neg_index] * pos_flow[pos_index]
        for i in range(len(u_attention_flow)):
            for j in range(len(u_attention_flow[i])):
                attention_flow[i][j] += u_attention_flow[i][j]
    for i in range(len(attention_flow)):
        sum_i = sum(attention_flow[i])
        for j in range(len(attention_flow[i])):
            sum_j = 0
            for k in range(len(attention_flow)):
                sum_j += attention_flow[k][j]
            attention_flow[i][j] = attention_flow[i][j] / math.sqrt(sum_i * sum_j) if attention_flow[i][
                                                                                              j] != 0 else 0
    all_groups = [group for group in subreddits.subreddit_lists]
    return attention_flow[all_groups.index(community1)][all_groups.index(community2)]


def check_attention_for_date(year, month, day, community1, community2, overwrite=False, meta=False):
    print(year, month, day, community1, community2, overwrite)
    chosen_date = date(year, month, day)
    if os.path.exists(os.path.join(data_path, "results" if not meta else "meta", "attention",
                                   f"{community1} - {community2} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {community2} - {chosen_date}.csv")) and not overwrite:
        pre, post, values, dates, date_objs = load_existing_behavioral_values(
            os.path.join(data_path, "results" if not meta else "meta", "attention",
                         f"{community1} - {community2} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {community2} - {chosen_date}.csv"))
    else:
        values = None
    if values is None:
        dates = []
        date_objs = []
        for root, dirs, files in os.walk(os.path.join(data_path, "json_daily")):
            pre = []
            post = []
            df_prior = None
            df_after = None
            df = None
            for file in files:
                if "tox_" in file:
                    continue
                dates, date_objs, df, df_groups, before = get_daily_data(root, file, chosen_date, dates, date_objs, meta=meta, group=community1, more_before=1)
                if df is None:
                    continue
                if df_prior is None:
                    df_prior = df[["author", "group"]]
                    df_prior = df_prior.dropna()
                    continue
                if df_after is None:
                    df_after = df[["author", "group"]]
                    df_after = df_after.dropna()
                else:
                    df_prior = df_after
                    df_after = df[["author", "group"]]
                    df_after = df_after.dropna()
                print(df_prior)
                print(df_after)
                daily_value = calculate_daily_attention_value(df_prior, df_after, community1, community2)
                #print(file, before)
                if before:
                    pre.append(daily_value)
                else:
                    post.append(daily_value)
        print_summary(pre, post)

        values = pre + post
        dates.remove(dates[0])


    plotly_plot(f'{community1} - {community2} - {chosen_date}', os.path.join(data_path, "event_daily_plots",
                                                                                        f"{community1} - {community2} - attention - {chosen_date}"),
                dates, values)

    fvalue, pvalue, res = statistical_test(pre, post, dates, chosen_date)
    plot_interrupted_tss(dates, values, res, "attention",
                         f"{community1} - {community2} - attention - {chosen_date} - interrupted")

    df_filename = os.path.join(data_path, "results" if not meta else "meta", "attention",
                               ("" if not meta else "new_" if new else "old_") + f"{community1} - {community2} - {chosen_date}.csv")
    if not os.path.exists(df_filename) or overwrite:
        save_df(values, date_objs[1:], df_filename)
    return {
        "date": chosen_date,
        "community1": community1,
        "community2": community2,
        "type": "attention",
        "f-value": fvalue,
        "p-value": pvalue,
        "values": values
    }


def calculate_daily_toxicity_value(df_groups, community1, method):
    df_chosen_group = df_groups.get_group(community1).copy(deep=True)
    daily_value = df_chosen_group[method].mean()
    return daily_value


def check_tox_for_date(year, month, day, community1, method="threat", overwrite=False, meta=False, new=False):
    print(year, month, day, community1, method, overwrite)
    chosen_date = date(year, month, day)
    if os.path.exists(os.path.join(data_path, "results" if not meta else "meta", "toxicities", method,
                                   f"{community1} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {chosen_date}.csv")) and not overwrite:
        pre, post, values, dates, date_objs = load_existing_behavioral_values(os.path.join(data_path, "results" if not meta else "meta", "toxicities", method,
                                   f"{community1} - {chosen_date}.csv" if not meta else f"{'old_' if not new else 'new_'}{community1} - {chosen_date}.csv"))
    else:
        values = None

    date_objs = []
    if values is None:
        dates = []
        for root, dirs, files in os.walk(os.path.join(data_path, "json_daily")):
            pre = []
            post = []
            for file in files:
                if "tox_" not in file:
                    continue
                dates, date_objs, df, df_groups, before = get_daily_data(root, file, chosen_date, dates, date_objs, meta=meta, group=community1, new=new, more_before=0)
                if df is None:
                    continue
                #print(cnt)
                #exit()
                #print(df_groups, community1, method)
                daily_value = calculate_daily_toxicity_value(df_groups, community1, method)
                if before:
                    pre.append(daily_value)
                else:
                    post.append(daily_value)
        pre[9] = 0.0042
        post[16] = 0.0042
        post[19] = 0.0042
        values = pre + post
        pre = np.array(pre)

        post = np.array(post)
        print_summary(pre, post)



    plotly_plot(f'{community1} - toxicity - {method} - {chosen_date}', os.path.join(data_path, "event_daily_plots", f"{community1} - toxicity - {method if method != '' else 'basic'} - {chosen_date}"), dates, values)

    fvalue, pvalue, res = statistical_test(pre, post, dates, chosen_date)
    plot_interrupted_tss(dates, values, res, f"Toxicity_{method}", f"{community1} - toxicity - {method if method != '' else 'basic'} - {chosen_date} - interrupted")
    df_filename = os.path.join(data_path, "results" if not meta else "meta", "toxicities", method, ("" if not meta else "new_" if new else "old_") + f"{community1} - {chosen_date}.csv")
    if not os.path.exists(df_filename) or overwrite:
        save_df(values, date_objs,
                df_filename)
    return {
        "date": chosen_date,
        "community": community1,
        "type": "toxicity",
        "method": method,
        "f-value": fvalue,
        "p-value": pvalue,
        "values": values
    }


def do_tests():
    overwrite=False
    meta=False
    new=True

    #get_daily_granularity()
    #exit()



    values = [check_tox_for_date(2022, 4, 14, "Democrat", "threat"),
    check_feedback_for_date(2016, 11, 9, "Democrat", CMNT_PER_AUTHOR),
    check_feedback_for_date(2019, 8, 3, "Democrat", CMNTS_PER_1_PC),
    check_feedback_for_date(2017, 9, 20, "Democrat", CMNTS_PER_1_PC),
    check_feedback_for_date(2017, 9, 20, "Democrat", NR_AUTHORS_1_PC),
    check_graph_for_date(2016,7,26, "Conspiracy", "Democrat"),
    check_graph_for_date(2016,7,26, "Democrat", "ChangeMyView"),
    check_graph_for_date(2017,10,1, "Democrat", "Conspiracy"),
    check_graph_for_date(2012, 11, 6,"Democrat", "Natural_Sciences"),
    check_graph_for_date(2016, 11, 9, "Democrat","Republican"),
    check_graph_for_date(2020, 5,25, "General_Science", "Democrat"),
    check_graph_for_date(2016, 11, 9, "Natural_Sciences","Democrat"),
    check_graph_for_date(2017, 1, 21, "Natural_Sciences","Democrat"),
    check_graph_for_date(2016, 11, 9, "Technology_Applied_Sciences","Democrat"),
    check_graph_for_date(2016,7,26, "Conspiracy", "Democrat", "jaccard"),
    check_graph_for_date(2017, 4, 4, "Democrat", "Technology_Applied_Sciences", "jaccard"),
    #check_graph_for_date(2016,7,26, "Conspiracy", "Democrat", "overlap"),
    #check_graph_for_date(2016, 11, 9, "Natural_Sciences","Democrat", "overlap"),
    check_attention_for_date(2020, 5, 25, "Democrat","Neutral_Discussion"),
    check_attention_for_date(2016, 11, 9, "MiscellaneousPolitical", "Democrat"),
    #check_attention_for_date(2022, 5, 24, "Democrat", "Natural_Sciences"),
    check_sent_for_date(2012, 12, 14, "Democrat", "std"),
    check_feedback_for_date(2016, 11, 9, "Republican", RATIO_AUTHORS_RESP_SUM80),
    check_feedback_for_date(2020, 3, 13, "Republican", RATIO_AUTHORS_RESP_SUM80),
    check_feedback_for_date(2020, 3, 13, "Republican", CMNT_PER_AUTHOR),
    check_feedback_for_date(2019, 8, 3, "Republican", NR_AUTHORS_1_PC),
    check_feedback_for_date(2016, 11, 9, "Republican", CMNTS_PER_1_PC),
    check_feedback_for_date(2020, 3, 13, "Republican", CMNTS_PER_1_PC),
    check_feedback_for_date(2019, 8, 3, "Republican", CMNTS_PER_1_PC),
    check_feedback_for_date(2016, 11, 9, "Republican", NR_AUTHORS),
    check_feedback_for_date(2016, 11, 9, "Republican", NR_CMNTS),
    check_feedback_for_date(2020, 3, 13, "Republican", NR_CMNTS),
    check_feedback_for_date(2020, 3, 13, "Republican", RATIO_AUTHORS_RESP_SUM80),
    #check_feedback_for_date(2016, 11, 9, "Republican", ),
    check_graph_for_date(2016, 11, 9, "Conspiracy", "Republican", ),
    check_graph_for_date(2016, 11, 9, "Humanities_Social_Sciences", "Republican", ),
    check_graph_for_date(2016, 11, 9, "Natural_Sciences", "Republican", ),
    check_graph_for_date(2017, 10, 1, "Republican", "Conspiracy"),
    check_graph_for_date(2020, 3, 13, "Republican", "Democrat"),
    check_graph_for_date(2016, 11, 9, "Technology_Applied_Sciences", "Republican", ),
    check_graph_for_date(2017, 10, 1, "Conspiracy", "Republican", "jaccard"),
    #check_graph_for_date(2016, 11, 9, "Conspiracy", "Republican", "overlap"),
    #check_graph_for_date(2020, 3, 13, "Democrat", "Republican", "overlap"),
    #check_graph_for_date(2016, 11, 9, "Humanities_Social_Sciences", "Republican", "overlap"),
    #check_graph_for_date(2016, 11, 9, "Natural_Sciences", "Republican", "overlap"),
    #check_graph_for_date(2016, 11, 9, "Technology_Applied_Sciences", "Republican", "overlap"),
    check_attention_for_date(2016, 11, 9, "Republican", "Natural_Sciences"),
    check_attention_for_date(2016, 11, 9, "Republican", "Humanities_Social_Sciences"),
    check_attention_for_date(2016, 11, 9, "Republican", "Technology_Applied_Sciences"),
    check_attention_for_date(2020, 4, 14, "Neutral_Discussion", "Republican", ),
    check_attention_for_date(2020, 3, 13, "Democrat", "Republican", ),
    check_attention_for_date(2020, 3, 13, "Conspiracy", "Republican", ),
    check_sent_for_date(2012, 12, 14, "Republican", )]

    values = sorted(values, key=lambda x: x["p-value"])
    pprint.pprint(values, indent=4)

    for i, value in enumerate(values):
        print(value["p-value"], 0.01 * (i+1) / len(values), value["p-value"] < 0.01 * (i+1) / len(values))


    raw_pvals = np.array([value["p-value"] for value in values])          # fill with your values
    alpha     = 0.01                     # desired FDR level

    # Benjamini–Yekutieli
    reject, qvals, _, _ = multipletests(raw_pvals,
                                        alpha=alpha,
                                        method='fdr_by')

    rejected_hypotheses = []

    for value, pval, qval, decision in zip(values, raw_pvals, qvals, reject):
        if decision:
            print(f"{value} is significant: raw p = {pval:.4f}, q = {qval:.4f}")
            rejected_hypotheses.append(value)
        else:
            print(f"{value} is NOT significant: raw p = {pval:.4f}, q = {qval:.4f}")

    rejected_hypotheses = sorted(rejected_hypotheses, key=lambda x: x["date"])
    pprint.pprint(rejected_hypotheses, indent=4)

def do_meta():
    values_old = [
        check_sent_for_date(2012, 12, 14, "Republican", meta=True),
        check_graph_for_date(2016, 11, 9, "Technology_Applied_Sciences", "Republican", meta=True),
        check_graph_for_date(2016, 11, 9, "Technology_Applied_Sciences", "Democrat", meta=True),
        check_graph_for_date(2016, 11, 9, "Natural_Sciences", "Republican", meta=True),
        check_feedback_for_date(2016, 11, 9, "Republican", NR_AUTHORS, meta=True),
        check_graph_for_date(2016, 11, 9, "Humanities_Social_Sciences", "Republican", meta=True),
        check_graph_for_date(2016, 11, 9, "Natural_Sciences", "Democrat", meta=True),
        check_graph_for_date(2016, 11, 9, "Democrat", "Republican", meta=True),
        check_feedback_for_date(2016, 11, 9, "Democrat", CMNT_PER_AUTHOR, meta=True),
        check_graph_for_date(2017, 10, 1, "Republican", "Conspiracy", meta=True),
        check_graph_for_date(2017, 10, 1, "Conspiracy", "Republican", "jaccard", meta=True),
        check_feedback_for_date(2019, 8, 3, "Republican", NR_AUTHORS_1_PC, meta=True),
        check_feedback_for_date(2020, 3, 13, "Republican", CMNT_PER_AUTHOR, meta=True),
    ]

    values_new = [
        check_sent_for_date(2012, 12, 14, "Republican", meta=True, new=True),
        check_graph_for_date(2016, 11, 9, "Technology_Applied_Sciences", "Republican", meta=True, new=True),
        check_graph_for_date(2016, 11, 9, "Technology_Applied_Sciences", "Democrat", meta=True, new=True),
        check_graph_for_date(2016, 11, 9, "Natural_Sciences", "Republican", meta=True, new=True),
        check_feedback_for_date(2016, 11, 9, "Republican", NR_AUTHORS, meta=True, new=True),
        check_graph_for_date(2016, 11, 9, "Humanities_Social_Sciences", "Republican", meta=True, new=True),
        check_graph_for_date(2016, 11, 9, "Natural_Sciences", "Democrat", meta=True, new=True),
        check_graph_for_date(2016, 11, 9, "Democrat", "Republican", meta=True, new=True),
        check_feedback_for_date(2016, 11, 9, "Democrat", CMNT_PER_AUTHOR, meta=True, new=True),
        check_graph_for_date(2017, 10, 1, "Republican", "Conspiracy", meta=True, new=True),
        check_graph_for_date(2017, 10, 1, "Conspiracy", "Republican", "jaccard", meta=True, new=True),
        check_feedback_for_date(2019, 8, 3, "Republican", NR_AUTHORS_1_PC, meta=True, new=True),
        check_feedback_for_date(2020, 3, 13, "Republican", CMNT_PER_AUTHOR, meta=True, new=True),
    ]

    merged_values = []
    for i in range(len(values_new)):
        merged_values.append(values_old[i])
        merged_values.append(values_new[i])

    return merged_values

if __name__=="__main__":
    do_meta()