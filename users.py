import collections

import pandas as pd

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
import pandas
import os

if __name__ == '__main__':
    users_groups = collections.defaultdict(set)
    df_all_users_groups = {}
    for root, dirs, files in os.walk(os.path.join(data_path, "user_group_counts_all")):
        for file in files:
            df = pandas.read_json(os.path.join(root, file))
            for group in df["group"].unique():
                print(file, group)
                if group not in df_all_users_groups:
                    if os.path.exists(os.path.join(data_path, f"{group}_users.csv")):
                        df_all_users_groups[group] = pd.read_csv(os.path.join(data_path, f"{group}_users.csv"))
                    else:
                        df_all_users_groups[group] = pd.DataFrame(columns=["author", "joined"])
                if os.path.splitext(file)[0] in df_all_users_groups[group]["joined"].values:
                    print(f"{group} for {os.path.splitext(file)[0]} already done")
                    continue
                df_group = df[df["group"] == group]
                group_users = set(df_group["author"].unique())
                new_users_group = group_users.difference(users_groups[group])
                users_groups[group].update(group_users)
                df_new_group = pd.DataFrame()
                df_new_group["author"] = list(new_users_group)
                df_new_group["joined"] = os.path.splitext(file)[0]
                df_all_users_groups[group] = pandas.concat([df_all_users_groups[group], df_new_group])
                df_all_users_groups[group].to_csv(os.path.join(data_path, f"{group}_users.csv"))

