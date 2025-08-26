import collections
import json
import math
import os
import pprint

import imageio
import seaborn as sns

from matplotlib import pyplot as plt
from numpy.linalg import norm
import pandas as pd

import subreddits

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
json_path = os.path.join(data_path, "json")
com_cnt = len(subreddits.subreddit_lists)

def get_dirs():
    dirs = os.listdir(json_path)
    return [f"{json_path}/{dir}" for dir in dirs]
def get_user_activity_per_group():
    dirs = get_dirs()
    nr_partitions = 16
    folder = "user_group_counts_all"
    os.makedirs(os.path.join(data_path, f"{folder}"), exist_ok=True)
    for directory in dirs:
        if os.path.exists(os.path.join(data_path, f"{folder}/{directory.split('/')[-1]}.json")):
            continue
        print(os.path.join(data_path, f"{folder}/{directory.split('/')[-1]}.json"))
        for root, folders, files in os.walk(directory):
            dfs = [pd.DataFrame(columns=["author", "group", "count"]) for i in range(nr_partitions)]
            for i, file in enumerate(files):
                if ".json" not in file:
                    continue
                print(root, file)
                file_name = os.path.join(root, file)
                df_part = pd.read_json(file_name, lines=True)
                df_part = df_part[["author", "subreddit"]]
                df_part["group"] = df_part["subreddit"].str.lower().map(subreddits.subreddit_to_group)
                df_part = df_part[["author", "group"]]
                df_part = df_part.dropna(subset=["group"])
                groups = df_part.groupby(by=["author", "group"]).size()
                group_sizes_df = groups.reset_index(name='count')
                if len(dfs[i % nr_partitions]) == 0:
                    dfs[i % nr_partitions] = group_sizes_df
                else:
                    merged_df = pd.merge(dfs[i % nr_partitions], group_sizes_df, on=['author', 'group'], how='outer', suffixes=('_df1', '_df2'))
                    merged_df['count'] = merged_df['count_df1'].fillna(0) + merged_df['count_df2'].fillna(0)
                    dfs[i % nr_partitions] = merged_df.drop(columns=['count_df1', 'count_df2'])
            df = pd.DataFrame(columns=["author", "group", "count"])
            for df_part in dfs:
                merged_df = pd.merge(df, df_part, on=['author', 'group'], how='outer', suffixes=('_df1', '_df2'))
                merged_df['count'] = merged_df['count_df1'].fillna(0) + merged_df['count_df2'].fillna(0)
                df = merged_df.drop(columns=['count_df1', 'count_df2'])
            df.to_json(os.path.join(data_path, f"{folder}/{directory.split('/')[-1]}.json"))


def C_matrix():
    folder = "user_group_counts_all"
    os.makedirs(os.path.join(data_path, "attention_flow_C"), exist_ok=True)
    for root, dirs, files in os.walk(os.path.join(data_path, f"{folder}")):
        users = set()
        if not os.path.exists(os.path.join(data_path, "all_users.json")):
            for file in files:
                print(file, 1)
                df = pd.read_json(os.path.join(root, file))
                users = users.union((df["author"].unique()))
            user_list = sorted(list(users))
            with open(os.path.join(data_path, "all_users.json"), "wt") as f:
                json.dump(user_list, f)
        else:
            with open(os.path.join(data_path, "all_users.json"), "rt") as f:
                #user_list = json.load(f)
                user_list = []
        C = []

        for matrix_index, file in enumerate(files):
            if not os.path.exists(os.path.join(data_path, "attention_flow_C", file)):

                print(file, 3)
                df = pd.read_json(os.path.join(root, file))
                C = {user: [0 for _ in range(com_cnt)] for user in df["author"].unique()}
                df_grouped = df.groupby("group")
                for g_index, group in df_grouped:
                    ind = [key for key in subreddits.subreddit_lists.keys()].index(g_index)
                    for r_index, row in group.iterrows():
                        C[row["author"]][ind] = row["count"]
                with open(os.path.join(data_path, "attention_flow_C", file), "wt") as f:
                    json.dump(C, f)

def B_matrix():
    c_folder = os.path.join(data_path, "attention_flow_C")
    b_folder = os.path.join(data_path, "attention_flow_B")
    os.makedirs(b_folder, exist_ok=True)
    for root, dirs, files in os.walk(c_folder):
        for i in range(len(files) - 1):
            c_matrices = []
            if not os.path.exists(os.path.join(b_folder, files[i + 1])):
                print(files[i])
                for j in range(i, i+2):
                    with open(os.path.join(root, files[j]), "rt") as f:
                        c_matrices.append(json.load(f))
                users = set()
                for j in range(len(c_matrices)):
                    if len(users) == 0:
                        users = set([user for user in c_matrices[j].keys()])
                    else:
                        users = users.intersection([user for user in c_matrices[j].keys()])
                    for key in c_matrices[j].keys():
                        sum_ = sum(c_matrices[j][key])
                        for k in range(len(c_matrices[j][key])):
                            c_matrices[j][key][k] = c_matrices[j][key][k] / sum_
                b_matrix = {user: [0 for com in range(com_cnt)] for user in sorted(list(users))}
                for user in list(users):
                    for j in range(com_cnt):
                        if c_matrices[0][user][j] == 0 or c_matrices[1][user][j] == 0:
                            b_matrix[user][j] = c_matrices[0][user][j] - c_matrices[1][user][j]
                        else:
                            b_matrix[user][j] = 0
                with open(os.path.join(b_folder, files[i + 1]), "wt") as f:
                    json.dump(b_matrix, f)


def attention_flow():
    b_folder = os.path.join(data_path, "attention_flow_B")
    F_folder = os.path.join(data_path, "attention_flow_F")
    os.makedirs(F_folder, exist_ok=True)
    for root, dirs, files in os.walk(b_folder):
        for file in files:
            attention_flow = [[0 for _ in range(com_cnt)] for __ in range(com_cnt)]
            with open(os.path.join(root, file), "rt") as f:
                b_matrix = json.load(f)
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
                neg_flow_normalized = norm(neg_flow,1)
                for i in range(len(neg_flow)):
                    neg_flow[i] = neg_flow[i] / neg_flow_normalized
                for pos_index, pos_flow_index in enumerate(pos_flow_indeces):
                    for neg_index, neg_flow_index in enumerate(neg_flow_indeces):
                        u_attention_flow[neg_flow_index][pos_flow_index] = neg_flow[neg_index] * pos_flow[pos_index]
                for i in range(len(u_attention_flow)):
                    for j in range(len(u_attention_flow[i])):
                        attention_flow[i][j] += u_attention_flow[i][j]
            with open(os.path.join(F_folder, file), "wt") as f:
                json.dump(attention_flow, f)

def attention_flow_normalized():
    F_folder = os.path.join(data_path, "attention_flow_F")
    F_normalized_folder = os.path.join(data_path, "attention_flow_F_normalized")
    os.makedirs(F_normalized_folder, exist_ok=True)
    for root, dirs, files in os.walk(F_folder):
        for file in files:
            with open(os.path.join(root, file), "rt") as f:
                attention_matrix = json.load(f)
            for i in range(len(attention_matrix)):
                sum_i = sum(attention_matrix[i])
                for j in range(len(attention_matrix[i])):
                    sum_j = 0
                    for k in range(len(attention_matrix)):
                        sum_j += attention_matrix[k][j]
                    attention_matrix[i][j] = attention_matrix[i][j] / math.sqrt(sum_i * sum_j) if attention_matrix[i][j] != 0 else 0
            with open(os.path.join(F_normalized_folder, file), "wt") as f:
                json.dump(attention_matrix, f)


def temporal_attention(suffix):
    folder = f"attention_flow_{suffix}"
    frames_dir = os.path.join(data_path, f"attention_viz_{suffix}")
    os.makedirs(frames_dir, exist_ok=True)
    frame_files = []
    for root, dirs, files in os.walk(os.path.join(data_path, folder)):
        for i, file in enumerate(files):
            with open(os.path.join(root, file), "rt") as f:
                matrix = json.load(f)
            plt.figure(figsize=(12, 10))
            com_names = [com for com in subreddits.subreddit_lists.keys()]
            for j in range(len(com_names)):
                com_names[j] = com_names[j].replace("_Discussion", "")
                com_names[j] = com_names[j].replace("_Sciences", "")
                com_names[j] = com_names[j].replace("Political", "")
                com_names[j] = com_names[j].replace("ChangeMyView", "CMV")
            sns.heatmap(matrix, cmap='viridis', annot=True, xticklabels=com_names, yticklabels=com_names)
            plt.xticks(rotation=45)
            plt.title(os.path.splitext(file)[0])
            frame_file = f'{frames_dir}/frame_{i}.png'
            plt.savefig(frame_file)
            plt.close()
            frame_files.append(frame_file)


    gif_path = os.path.join(data_path, f"{frames_dir}/network_evolution.gif")
    with imageio.get_writer(gif_path, mode='I', duration=1) as writer:
        for filename in frame_files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def attention_time_series(suffix):
    frames_dir = os.path.join(data_path, f"attention_viz_{suffix}")
    folder = f"attention_flow_{suffix}"
    os.makedirs(frames_dir, exist_ok=True)
    time_series_dict = collections.defaultdict(list)
    for root, dirs, files in os.walk(os.path.join(data_path, folder)):
        for i, file in enumerate(files):
            with open(os.path.join(root, file), "rt") as f:
                matrix = json.load(f)
            com_names = [com for com in subreddits.subreddit_lists.keys()]
            for j, com1 in enumerate(com_names):
                for k, com2 in enumerate(com_names):
                    if j == k:
                        continue
                    if com1 not in ["Republican", "Democrat"] and com2 not in ["Republican", "Democrat"]:
                        continue
                    time_series_dict[f"{com1} - {com2}"].append(matrix[j][k])
    df = pd.DataFrame(time_series_dict, index=pd.DatetimeIndex(pd.date_range(pd.Timestamp("2012-02"), pd.Timestamp("2022-12"), freq="MS")))
    df.to_csv(os.path.join(data_path, f"attention_flow_{suffix}"))


for suffix in ["F", "F_normalized"]:
    temporal_attention(suffix)