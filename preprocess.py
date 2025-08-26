import os
import subprocess

import pandas as pd

import subreddits

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
json_path = os.path.join(data_path, "json")
json_filtered_path = os.path.join(data_path, "json_filtered")
json_filtered_active_path = os.path.join(data_path, "json_filtered_active")
path_to_files = r"C:\Users\csabs\Downloads\2022-03\2022-03"
def get_body():
    os.makedirs(json_filtered_path, exist_ok=True)
    for root, dirs, files in os.walk(json_path):
        if "2022" not in root:
            continue
        for file in files:
            os.makedirs(root.replace(json_path, json_filtered_path), exist_ok=True)
            file_path = os.path.join(root, file)
            new_file_path = os.path.join(root, file).replace(json_path, json_filtered_path).replace(".json", ".parquet")
            if os.path.exists(new_file_path):
                continue
            if file.endswith(".json"):
                try:
                    df = pd.read_json(file_path)
                except:
                    df = pd.read_json(file_path, lines=True)
                df["group"] = df["subreddit"].str.lower().map(subreddits.subreddit_to_group)
                df = df[df["group"].isin(["Democrat", "Republican"])]
                df = df[["id", "body", "group"]]
                print(file_path.replace(json_path, json_filtered_path))
                df.to_parquet(new_file_path)


def merge_files():
    os.makedirs(json_filtered_active_path, exist_ok=True)
    for root, dirs, files in os.walk(json_filtered_path):
        df = pd.DataFrame()
        if os.path.exists(root.replace(json_filtered_path, json_filtered_active_path) + ".parquet"):
            continue
        print(root)
        for file in files:
            print(file)
            if "2021" not in root:
                continue
            df_part = pd.read_parquet(os.path.join(root, file))
            if len(df) == 0:
                df = df_part
            else:
                df = pd.concat([df, df_part])
        if len(df) > 0:
            df = df[df["body"].str.len() > 3]
            df.to_parquet(root.replace(json_filtered_path, json_filtered_active_path) + ".parquet")

def extract_2022_03():
    for root, dirs, files in os.walk(path_to_files):
        num_partitions = 16
        dfs = [pd.DataFrame() for _ in range(num_partitions)]
        for i, file in enumerate(files):
            print(file)
            curr_part = i % num_partitions
            df_part = pd.read_parquet(os.path.join(root, file))
            if len(dfs[curr_part]) == 0:
                dfs[curr_part] = df_part
            else:
                df = pd.concat([dfs[curr_part], df_part])
        df = pd.DataFrame()
        for df_i in dfs:
            df = pd.concat(df, df_i)

if __name__ == '__main__':
    #get_body()
    merge_files()



