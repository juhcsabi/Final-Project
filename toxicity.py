import collections
import os
import subprocess
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from detoxify import Detoxify
from collections import defaultdict
import plotly.graph_objects as go

device = "cpu"
data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"


def plotly_tox_ts(df, key):
    output_folder = os.path.join(data_path, f"toxicity_ts")
    os.makedirs(output_folder, exist_ok=True)
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')
    os.makedirs(output_folder, exist_ok=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=df["Mean"].values,
        mode='lines+markers',
        name=f'{df["Group"].values[0]}'
    ))

    fig.update_layout(
        title=f'{df["Group"].values[0]} - {key}',
        xaxis=dict(title="Time"),
        yaxis=dict(title='Weight'),
        width=1500,
        height=900
    )

    fig.show()
    fig.write_html(os.path.join(data_path, f"{output_folder}/{df['Group'].values[0]}_{key}.html"))

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def tox_analyze_group(df, batch_size=256):
    start_time = time.time()
    results = []

    # Check for duplicates
    unique_texts = list(set(df["body"].to_list()))
    text_to_indices = defaultdict(list)
    for idx, text in enumerate(df["body"]):
        text_to_indices[text].append(idx)

    print(f"Processing {len(df)} entries with batch size {batch_size}")
    predictor = Detoxify('original', device=device)
    dataset = TextDataset(unique_texts)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    unique_results = []
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch_start_time = time.time()
                preds = predictor.predict(batch)
                unique_results.extend(preds["toxicity"])
                batch_duration = time.time() - batch_start_time
                print(f"Batch {batch_idx + 1}/{len(data_loader)} completed in {batch_duration:.2f} seconds")
    except Exception as e:
        print("Exception occurred:", e)
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    indeces = []

    # Map the results back to the original indices
    for text, toxicity in zip(unique_texts, unique_results):
        for idx in text_to_indices[text]:
            results.append(toxicity)

    duration = time.time() - start_time
    print(f"Completed in {duration:.2f} seconds")

    return results


def toxicity(year, month, subsample_size):
    start_time = time.time()
    os.makedirs("csv", exist_ok=True)
    month_str = f"{month:02d}"
    df = pd.read_parquet(f"json_filtered_active/{year}-{month_str}.parquet")

    df_dem = df[df["group"] == "Democrat"]
    df_rep = df[df["group"] == "Republican"]

    if subsample_size == 0:
        subsample_size = min(len(df_dem), len(df_rep))
    else:
        subsample_size = min(min(len(df_dem), len(df_rep)), subsample_size)

    df_dem = df_dem.sample(subsample_size)
    df_rep = df_rep.sample(subsample_size)

    print(f"Analyzing {year}-{month_str} Democrat group")
    df_dem["toxicity"] = tox_analyze_group(df_dem)
    df_dem.to_csv(f"csv/{year}-{month_str}_d.csv")

    print(f"Analyzing {year}-{month_str} Republican group")
    df_rep["toxicity"] = tox_analyze_group(df_rep)
    df_rep.to_csv(f"csv/{year}-{month_str}_r.csv")

    duration = time.time() - start_time
    print(f"Completed processing {year}-{month_str} in {duration:.2f} seconds")
    return subsample_size


def toxicity_all():
    subsample_size = 0
    for year in range(2012, 2022):
        for month in range(1, 13):
            command = ["aws", "s3", "cp", f"s3://reddit-juhcsabi/{year}-{month:02d}.parquet", f"json_filtered_active/{year}-{month:02d}.parquet"]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print("Command executed successfully")
                print(result.stdout)
                subsample_size = toxicity(year, month, subsample_size)
            else:
                print("Command failed", command)
                print(result.stderr)


#toxicity_all()
def mean_toxicities_type(key):
    toxicity_means = collections.defaultdict(dict)
    for root, dirs, files in os.walk(os.path.join(data_path, "csv")):
        for file in files:
            filename = os.path.splitext(file)[0]
            date, group = filename.split("_")
            year, month = date.split("-")
            print(os.path.join(root, file))
            df = pd.read_csv(os.path.join(root, file), low_memory=False, encoding="utf-8", lineterminator='\n')
            if key not in df.columns:
                continue
            toxicity_means[group][year, month] = df[key].mean()
    abbr_to_group = {"r": "Republican", "d": "Democrat"}
    df = pd.DataFrame(columns=["Group", "Year", "Month", "Mean"])
    for group, means_dict in toxicity_means.items():
        rows = []
        for (year, month), mean in means_dict.items():
            print(group, (year, month), mean)
            rows.append([abbr_to_group[group], year, month, mean])
        df_group = pd.DataFrame(rows, columns=["Group", "Year", "Month", "Mean"])
        plotly_tox_ts(df_group, key)
        df = pd.concat([df, df_group])
    df.to_csv(os.path.join(data_path, "toxicities", f"{key}.csv"))


def mean_toxicities():
    for key in ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]:
        mean_toxicities_type(key)

def all_toxicities_plotly():
    fig = go.Figure()
    dates = pd.date_range(start='2012-01-01', end='2022-12-31', freq='MS')

    for key in ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]:
        df = pd.read_csv(os.path.join(data_path, "toxicities", f"{key}.csv"))
        values_r = df[df["Group"] == "Republican"]["Mean"].values
        values_d = df[df["Group"] == "Democrat"]["Mean"].values
        fig.add_trace(go.Scatter(
            x=dates,
            y=values_r,
            mode='lines+markers',
            name=f'Republican - {key}'
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=values_d,
            mode='lines+markers',
            name=f'Democrat - {key}'
        ))

    fig.update_layout(
        title=f'Toxicities',
        xaxis=dict(title="Time"),
        yaxis=dict(title='Weight'),
        width=1500,
        height=900
    )
    fig.show()
    fig.write_html(os.path.join(data_path, f"toxicities/all.html"))
