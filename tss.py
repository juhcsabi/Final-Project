import collections
import os.path
import pprint

import networkx
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL, seasonal_decompose

import subreddits

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"


def plot_time_series(original, normalized, edge1, edge2, diff=False):
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq="MB")

    plt.figure(figsize=(12, 6))
    plt.plot(dates, original, label='Original')
    plt.plot(dates, normalized, label='Normalized', linestyle='--')
    plt.legend()
    plt.title(f'{edge1} - {edge2} {f"normalized on {edge1}" if diff else ""}')
    plt.show()

def detrend_self(deseasonalized_series, fitted):
    trend_component = fitted.trend
    detrended_series = deseasonalized_series - trend_component
    return detrended_series


def rescale_series(series_to_rescale, target_series):
    """
    Rescale a series to match the scale of a target series using MinMax scaling.

    Parameters:
    series_to_rescale (pd.Series): The series to be rescaled.
    target_series (pd.Series): The target series for scaling.

    Returns:
    rescaled_series (pd.Series): The rescaled series.
    """
    # scaler = MinMaxScaler()
    original_index = series_to_rescale.index
    #
    series_to_rescale = series_to_rescale.values.reshape(-1, 1)
    target_series = target_series.values.reshape(-1, 1)
    #
    # scaler.fit(target_series)
    # rescaled_series = scaler.transform(series_to_rescale).flatten()
    # print(target_series)
    # print(series_to_rescale)
    #
    # return pd.Series(rescaled_series, index=original_index)
    scaling_factor = target_series.std() / series_to_rescale.std()

    # Rescale the trend component
    rescaled_trend = series_to_rescale * scaling_factor

    return pd.Series(np.squeeze(rescaled_trend), index=original_index)


def detrend_using_reference_trend(original_series, reference_series, period):
    reference_trend = pd.Series(seasonal_decompose(reference_series, period=period).trend)

    # Step 2: Rescale the reference trend to the scale of the original series
    rescaled_trend = rescale_series(reference_trend, pd.Series(original_series))

    # Step 3: Detrend the original series by subtracting the rescaled trend
    detrended_series = original_series - rescaled_trend

    return pd.Series(detrended_series), pd.Series(rescaled_trend)

# Sample usage
def tsa_graph(sim_type=None):
    if sim_type is None:
        sim_type = ''
    else:
        sim_type = "_" + sim_type
    folder = f"graphs_group{sim_type}"
    edge_series = collections.defaultdict(list)
    for root, dirs, files in os.walk(os.path.join(data_path, folder)):
        for file in files:
            G = networkx.read_graphml(os.path.join(root, file))
            for edge1, edge2, data in G.edges(data=True):
                if sim_type == "":
                    edge = (edge1, edge2)
                else:
                    edge = (edge1, edge2) if len(edge_series[(edge1, edge2)]) >= len(edge_series[(edge2, edge1)]) else (edge2, edge1)
                edge_series[edge].append((data["weight"], os.path.splitext(file)[0]))
    for (edge1, edge2) in edge_series:
        edge_series[(edge1, edge2)].sort(key=lambda x: (int(x[1].split("-")[0]), int(x[1].split("-")[1])))

    folder = "user_group_counts"
    size_series = collections.defaultdict(list)
    for root, dirs, files in os.walk(os.path.join(data_path, folder)):
        for file in files:
            df = pandas.read_json(os.path.join(root, file))
            df_group_counts = df.groupby("group").count()
            for group, count in df_group_counts.iterrows():
                size_series[group].append((count["count"], os.path.splitext(file)[0]))
    for key in size_series:
        size_series[key].sort(key=lambda x: (int(x[1].split("-")[0]), int(x[1].split("-")[1])))

    groups_examined = ["Democrat", "Republican"]
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')
    for (edge1, edge2), weight_year_list in edge_series.items():

        if edge1 not in groups_examined and edge2 not in groups_examined:
            continue
        weight_list = [weight for (weight, year) in weight_year_list]
        if len(weight_list) == 0:
            continue
        cmv1 = (edge1 == "ChangeMyView")
        cmv2 = (edge2 == "ChangeMyView")

        result = seasonal_decompose(weight_list, period=12)
        seasonal_component = result.seasonal
        deseasonalized_series = np.array(weight_list) - seasonal_component
        print(deseasonalized_series)
        detrended = detrend_self(deseasonalized_series, result)
        print(result.trend)
        df = pd.DataFrame({
            'dates': dates if not (cmv1 or cmv2) else pd.date_range(start='2013-01-01', end='2022-12-31', freq='MS'),
            'weights': weight_list ,
            'seasonal_component': seasonal_component,
            'detrended': detrended,
            'trend': result.trend
        })
        tss_path = os.path.join(data_path, f"time_series{sim_type}")
        os.makedirs(tss_path, exist_ok=True)
        df.to_csv(os.path.join(tss_path, f"{edge1} - {edge2}.csv"))

def tsa_graph_plotly(sim_type=None):
    if sim_type is None:
        sim_type = ''
    else:
        sim_type = "_" + sim_type
    tss_path = os.path.join(data_path, f"time_series{sim_type}")
    os.makedirs(tss_path, exist_ok=True)
    output_folder = os.path.join(data_path, f"tsa_plotly{sim_type}")
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='ME')
    os.makedirs(output_folder, exist_ok=True)
    series = collections.defaultdict(dict)
    for root, dirs, files in os.walk(tss_path):
        for file in files:
            df = pd.read_csv(os.path.join(root, file))
            for column in df.columns:
                if column != "dates":
                    if "Change" in file:
                        lst = [0 for i in range(12)]
                        lst.extend(df[column].values)
                        series[column][os.path.splitext(file)[0]] = lst
                    else:
                        series[column][os.path.splitext(file)[0]] = df[column].values


    # Choose specific edges to plot or plot all
    columns_to_plot = list(series.keys())  # Modify this as needed


    for column in columns_to_plot:
        fig = go.Figure()

        for edge in series[column]:
            weights = series[column][edge]
            fig.add_trace(go.Scatter(
                x=dates,
                y=weights,
                mode='lines+markers',
                name=f'{edge}'
            ))

            fig.update_layout(
                title=column + sim_type,
                xaxis=dict(title="Time"),
                yaxis=dict(title='Weight'),
                width=1500,
                height=900
            )

            # Show the figure
        fig.show()
        fig.write_html(os.path.join(data_path, f"{output_folder}/{column}.html"))

def tsa_fb():
    attr_series_dem = collections.defaultdict(list)
    attr_series_rep = collections.defaultdict(list)
    df = pd.read_csv(os.path.join(data_path, "community_feedback.csv"))
    df_dem = df[df["Group"] == "Democrat"]
    df_rep = df[df["Group"] == "Republican"]
    for column in df.columns:
        attr_series_dem[column] = df_dem[column].to_list()
        attr_series_rep[column] = df_rep[column].to_list()

    folder = f"tss_community_feedback"
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')
    for attr_series in [attr_series_dem, attr_series_rep]:
        for attr, values in attr_series.items():
            if attr == "Group" or attr == "Year" or attr == "Month":
                continue
            result = seasonal_decompose(values, period=12)
            seasonal_component = result.seasonal
            deseasonalized_series = np.array(values) - seasonal_component
            print(deseasonalized_series)
            detrended = detrend_self(deseasonalized_series, result)
            print(result.trend)
            df = pd.DataFrame({
                'dates': dates,
                'weights': values,
                'seasonal_component': seasonal_component,
                'detrended': detrended,
                'trend': result.trend
            })
            tss_path = os.path.join(data_path, folder)
            os.makedirs(tss_path, exist_ok=True)
            df.to_csv(os.path.join(tss_path, f"{attr_series['Group'][0]}_{attr}.csv"))

def tsa_fb_plotly():
    tss_path = os.path.join(data_path, f"tss_community_feedback")
    os.makedirs(tss_path, exist_ok=True)
    output_folder = os.path.join(data_path, f"tss_community_feedback_plotly")
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')
    os.makedirs(output_folder, exist_ok=True)
    series = collections.defaultdict(dict)
    for root, dirs, files in os.walk(tss_path):
        for file in files:
            df = pd.read_csv(os.path.join(root, file))
            for column in df.columns:
                if column != "dates":
                    if "Change" in file:
                        lst = [0 for i in range(12)]
                        lst.extend(df[column].values)
                        series[column][os.path.splitext(file)[0]] = lst
                    else:
                        series[column][os.path.splitext(file)[0]] = df[column].values
    columns_to_plot = list(series.keys())  # Modify this as needed


    for column in columns_to_plot:
        fig = go.Figure()
        for edge in series[column]:
            weights = series[column][edge]
            fig.add_trace(go.Scatter(
                x=dates,
                y=weights,
                mode='lines+markers',
                name=f'{edge}'
            ))

            fig.update_layout(
                title=column,
                xaxis=dict(title="Time"),
                yaxis=dict(title='Weight'),
                width=1500,
                height=900
            )

            # Show the figure
        fig.show()
        fig.write_html(os.path.join(data_path, f"{output_folder}/{column}.html"))


def tsa_tox():
    attr_series_dem = collections.defaultdict(list)
    attr_series_dem["Group"].append("Democrat")
    attr_series_rep = collections.defaultdict(list)
    attr_series_rep["Group"].append("Republican")

    for key in ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]:
        df = pd.read_csv(os.path.join(data_path, "toxicities", f"{key}.csv"))
        df_dem = df[df["Group"] == "Democrat"]
        df_rep = df[df["Group"] == "Republican"]
        attr_series_dem[key] = df_dem["Mean"].to_list()
        attr_series_rep[key] = df_rep["Mean"].to_list()

    folder = f"tss_toxicities"
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')

    for attr_series in [attr_series_dem, attr_series_rep]:
        for attr, values in attr_series.items():
            if attr == "Group":
                continue
            result = seasonal_decompose(values, period=12)
            seasonal_component = result.seasonal
            deseasonalized_series = np.array(values) - seasonal_component
            print(deseasonalized_series)
            detrended = detrend_self(deseasonalized_series, result)
            print(result.trend)
            df = pd.DataFrame({
                'dates': dates,
                'weights': values,
                'seasonal_component': seasonal_component,
                'detrended': detrended,
                'trend': result.trend
            })
            tss_path = os.path.join(data_path, folder)
            os.makedirs(tss_path, exist_ok=True)
            df.to_csv(os.path.join(tss_path, f"{attr_series['Group'][0]}_{attr}.csv"))


def tsa_tox_plotly():
    tss_path = os.path.join(data_path, f"tss_toxicities")
    os.makedirs(tss_path, exist_ok=True)
    output_folder = os.path.join(data_path, f"tss_toxicities_plotly")
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')
    os.makedirs(output_folder, exist_ok=True)
    series = collections.defaultdict(dict)
    for root, dirs, files in os.walk(tss_path):
        for file in files:
            df = pd.read_csv(os.path.join(root, file))
            for column in df.columns:
                if column != "dates":
                    series[column][os.path.splitext(file)[0]] = df[column].values
    columns_to_plot = list(series.keys())  # Modify this as needed


    for column in columns_to_plot:
        fig = go.Figure()
        for edge in series[column]:
            weights = series[column][edge]
            fig.add_trace(go.Scatter(
                x=dates,
                y=weights,
                mode='lines+markers',
                name=f'{edge}'
            ))

            fig.update_layout(
                title=column,
                xaxis=dict(title="Time"),
                yaxis=dict(title='Weight'),
                width=1500,
                height=900
            )

            # Show the figure
        fig.show()
        fig.write_html(os.path.join(data_path, f"{output_folder}/{column}.html"))


def tsa_bg():
    attr_series_dem = collections.defaultdict(list)
    attr_series_rep = collections.defaultdict(list)
    df = pd.read_csv(os.path.join(data_path, "community_feedback.csv"))
    df_dem = df[df["Group"] == "Democrat"]
    df_rep = df[df["Group"] == "Republican"]
    for column in df.columns:
        attr_series_dem[column] = df_dem[column].to_list()
        attr_series_rep[column] = df_rep[column].to_list()

    folder = f"tss_community_feedback"
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')
    for attr_series in [attr_series_dem, attr_series_rep]:
        for attr, values in attr_series.items():
            if attr == "Group" or attr == "Year" or attr == "Month":
                continue
            result = seasonal_decompose(values, period=12)
            seasonal_component = result.seasonal
            deseasonalized_series = np.array(values) - seasonal_component
            print(deseasonalized_series)
            detrended = detrend_self(deseasonalized_series, result)
            print(result.trend)
            df = pd.DataFrame({
                'dates': dates,
                'weights': values,
                'seasonal_component': seasonal_component,
                'detrended': detrended,
                'trend': result.trend
            })
            tss_path = os.path.join(data_path, folder)
            os.makedirs(tss_path, exist_ok=True)
            df.to_csv(os.path.join(tss_path, f"{attr_series['Group'][0]}_{attr}.csv"))

def tsa_bg_plotly():
    tss_path = os.path.join(data_path, f"tss_community_feedback")
    os.makedirs(tss_path, exist_ok=True)
    output_folder = os.path.join(data_path, f"tss_community_feedback_plotly")
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')
    os.makedirs(output_folder, exist_ok=True)
    series = collections.defaultdict(dict)
    for root, dirs, files in os.walk(tss_path):
        for file in files:
            df = pd.read_csv(os.path.join(root, file))
            for column in df.columns:
                if column != "dates":
                    if "Change" in file:
                        lst = [0 for i in range(12)]
                        lst.extend(df[column].values)
                        series[column][os.path.splitext(file)[0]] = lst
                    else:
                        series[column][os.path.splitext(file)[0]] = df[column].values
    columns_to_plot = list(series.keys())  # Modify this as needed


    for column in columns_to_plot:
        fig = go.Figure()
        for edge in series[column]:
            weights = series[column][edge]
            fig.add_trace(go.Scatter(
                x=dates,
                y=weights,
                mode='lines+markers',
                name=f'{edge}'
            ))

            fig.update_layout(
                title=column,
                xaxis=dict(title="Time"),
                yaxis=dict(title='Weight'),
                width=1500,
                height=900
            )

            # Show the figure
        fig.show()
        fig.write_html(os.path.join(data_path, f"{output_folder}/{column}.html"))

