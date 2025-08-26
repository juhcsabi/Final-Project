import os

import networkx
import pandas as pd
from plotly import graph_objects as go
import subreddits

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
json_path = os.path.join(data_path, "json")

def get_dirs():
    dirs = os.listdir(json_path)
    return [f"{json_path}/{dir}" for dir in dirs]


def filter_top_1_percent(group):
    group = group.sort_values('score', ascending=False)
    top_1_percent_count = max(1, int(len(group) * 0.01))  # Calculate top 1%, at least one row
    return group.head(top_1_percent_count)


def get_user_activity_per_group():
    df_existing = pd.read_csv(os.path.join(data_path, "community_feedback.csv"))
    groups_to_check = ["Democrat", "Republican"]
    dirs = get_dirs()
    df_all = pd.DataFrame(columns=["Group", "Year", "Month",
                                                "Number of comments", "Number of authors", "Avg. comments per author",
                                                "Sum of scores", "Authors responsible for sum 80%", "Ratio of authors responsible for sum 80%",
                                                "Nr. of authors responsible for top 1%", "Comments per author responsible for top 1%"])
    for directory in dirs:
        for root, folders, files in os.walk(directory):
            df = pd.DataFrame(columns=["author", "subreddit", "score"])
            year_month = directory.split('/')[-1]
            if len(df_existing[(df_existing["Year"] == int(year_month.split('-')[0])) & (df_existing["Month"] == int(year_month.split('-')[1]))]) > 0:
                continue
            print(year_month)
            for file in files:
                if not file.endswith('.json'):
                    continue
                file_name = os.path.join(root, file)
                df_part = pd.read_json(file_name, lines=True)
                try:
                    df_part = df_part[["author", "subreddit", "score"]]
                except:
                    print(file, df_part)
                    exit()
                df_part["group"] = df_part["subreddit"].str.lower().map(subreddits.subreddit_to_group)
                df_part = df_part[df_part["group"].isin(groups_to_check)]
                #df_part = df_part.dropna(subset=["group"])
                if len(df) == 0:
                    df = df_part
                else:
                    df = pd.concat([df, df_part], ignore_index=True)
            for group in groups_to_check:
                df_subgroup = df[df["group"] == group].sort_values(by="score", ascending=False)
                print(year_month, group)
                print("Number of comments", len(df_subgroup))
                print("Number of unique users", len(df_subgroup["author"].unique()))
                print("Avg. comments per user", len(df_subgroup) / len(df_subgroup["author"].unique()))
                curr_sum = 0
                curr_count = 0
                score_sum = df_subgroup["score"].sum()
                for index, row in df_subgroup.iterrows():
                    curr_sum += row["score"]
                    curr_count += 1
                    if curr_sum >= 0.8 * score_sum:
                        break
                top_80_percent_score_sum_df = df_subgroup.head(curr_count)
                print("Sum of all scores", score_sum)
                print("Number of unique users responsible for 80% of all scores", len(top_80_percent_score_sum_df["author"].unique()))
                print("Ratio of users responsible for 80% of all scores", len(top_80_percent_score_sum_df["author"].unique()) / len(df_subgroup["author"].unique()))
                top_1_percent = df.head(max(1, int(len(df_subgroup) * 0.01)))
                print("Number of users responsible for the top 1% upvoted comments", len(top_1_percent["author"].unique()))
                print("Count of top 1% upvoted comments", len(top_1_percent))
                print("Comments per user in the top 1%", len(top_1_percent) / len(top_1_percent["author"].unique()))
                df_data = pd.DataFrame(columns=["Group", "Year", "Month",
                                                "Number of comments", "Number of authors", "Avg. comments per author",
                                                "Sum of scores", "Authors responsible for sum 80%", "Ratio of authors responsible for sum 80%",
                                                "Nr. of authors responsible for top 1%", "Comments per author responsible for top 1%"],
                                       data=[(group, year_month.split('-')[0], year_month.split('-')[1],
                                              len(df_subgroup), len(df_subgroup["author"].unique()), len(df_subgroup)/len(df_subgroup["author"].unique()),
                                              score_sum, len(top_80_percent_score_sum_df["author"].unique()), len(top_80_percent_score_sum_df["author"].unique()) / len(df_subgroup["author"].unique()),
                                              len(top_1_percent["author"].unique()), len(top_1_percent) / len(top_1_percent["author"].unique()))])
                df_all = pd.concat([df_all, df_data])
    df_all = pd.concat([df_existing, df_all])
    df_all = df_all.sort_values(by=["Year", "Month"])
    df_all.round(2).to_csv(os.path.join(data_path, "community_feedback.csv"), index=False)



def visualize():
    df = pd.read_csv(os.path.join(data_path, "community_feedback.csv"))
    for column in df.columns:
        if column in ["Year", "Month", "Group"]:
            continue
        df_type = df[[column, "Year", "Month", "Group"]]
        fig = go.Figure()
        for community in ["Democrat", "Republican"]:
            df_type_community = df_type[df_type["Group"] == community]
            fig.add_trace(go.Scatter(
                x=[f"{year}-{month}" for year, month in zip(df_type_community["Year"].values, df_type_community["Month"].values)],
                y=df_type_community[column],
                mode='lines+markers',
                name=f"{community} {column}"
            ))
        fig.update_layout(
            title=column,
            width=1500,
            height=900
        )
        fig.write_html(os.path.join(data_path, f"community_feedback_{column[:10]}.html"))
        fig.show()

def normalize_overlaps(on="group", sim_type=None):
    if sim_type is None or sim_type == "directed":
        graph_folder = f"graphs_{on}"
        graph_type = networkx.DiGraph
    else:
        graph_folder = f"graphs_{on}_{sim_type}"
    df = pd.read_csv(os.path.join(data_path, "community_feedback.csv"))
    df = df[["Year", "Month", "Group", "Number of authors"]]
    primaries = ["Democrat", "Republican"]
    df_ratio = pd.DataFrame(columns=["Year", "Month", "Primary", "Secondary", "Ratio"])
    for index, row in df.iterrows():
        G = networkx.Graph(networkx.read_graphml(os.path.join(data_path, graph_folder, f"{row['year']}-{row['month']}.graphml")))


if __name__ == '__main__':
    groups_to_check = ["Democrat", "Republican"]
    dirs = get_dirs()
    for directory in dirs:
        for root, folders, files in os.walk(directory):
            df = pd.DataFrame(columns=["id", "parent_id", "sentiment_score", "toxicity", "parent_sentiment_score", "parent_toxicity"])
            year_month = directory.split('/')[-1]
            for file in files:
                file_name = os.path.join(root, file)
                df_main = pd.read_json(file_name, lines=True)
                df_main = df_main[["id", "parent_id"]]
                df_main["group"] = df_main["subreddit"].str.lower().map(subreddits.subreddit_to_group)
                df_main = df_main[df_main["group"].isin(groups_to_check)]
                df_main = df_main.dropna(subset=["group"])
                df_nlp = pd.read_json(os.path.join(root.replace("json", "sentiment"), file))
                df_joined = df_main.merge(df_nlp, left_on="parent_id", right_on="id", how="inner")

                # df_part = df_part.dropna(subset=["group"])
                if len(df) == 0:
                    df = df_part
                else:
                    df = pd.concat([df, df_part], ignore_index=True)
