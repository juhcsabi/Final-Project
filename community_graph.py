import collections
from datetime import datetime
import os
import pprint

import subreddits
import networkx
import numpy as np
import pandas
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
import imageio

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
json_path = os.path.join(data_path, "json")



def get_dirs():
    dirs = os.listdir(json_path)
    return [f"{json_path}/{dir}" for dir in dirs]


def get_user_activity_per_subreddit():
    dirs = get_dirs()
    for directory in dirs:
        for root, folders, files in os.walk(directory):
            df = pd.DataFrame(columns=["author", "subreddit", "count"])
            for file in files:
                file_name = os.path.join(root, file)
                df_part = pd.read_json(file_name, lines=True)
                groups = df_part.groupby(by=["author", "subreddit"]).size()
                group_sizes_df = groups.reset_index(name='count')
                if len(df) == 0:
                    df = group_sizes_df
                else:
                    merged_df = pd.merge(df, group_sizes_df, on=['author', 'subreddit'], how='outer', suffixes=('_df1', '_df2'))
                    merged_df['count'] = merged_df['count_df1'].fillna(0) + merged_df['count_df2'].fillna(0)
                    df = merged_df.drop(columns=['count_df1', 'count_df2'])
            df["count"] = df["count"].astype(int)
            df = df[df["count"] >= int(df["count"].value_counts().median())]
            df.to_json(os.path.join(data_path, f"user_subreddit_counts/{directory.split('/')[-1]}.json"))

def get_user_activity_per_group():
    dirs = get_dirs()
    nr_partitions = 8
    for directory in dirs:
        if os.path.exists(os.path.join(data_path, f"user_group_counts/{directory.split('/')[-1]}.json")):
            continue
        print(os.path.join(data_path, f"user_group_counts/{directory.split('/')[-1]}.json"))
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
            df["count"] = df["count"].astype(int)
            df = df[df["count"] >= int(df["count"].value_counts().median())]
            df.to_json(os.path.join(data_path, f"user_group_counts/{directory.split('/')[-1]}.json"))


def get_overlaps(on="group", allow_self=False, overwrite=False):
    for root, dirs, files in os.walk(os.path.join(data_path, f"user_{on}_counts")):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.DataFrame(pd.read_json(file_path))
            communities = df[on].unique()
            folder = os.path.splitext(os.path.join(os.path.join(data_path, f"{on}_overlaps", file)))[0]
            print(folder)
            os.makedirs(folder, exist_ok=True)
            groups = df.groupby(on)
            for community in communities:
                if not overwrite and os.path.exists(f"{os.path.join(folder, community)}.json"):
                    continue
                community_users = groups.get_group(community)["author"].unique()
                if not allow_self:
                    filtered_df = df[(df["author"].isin(community_users)) & (df[on] != community)]
                else:
                    filtered_df = df[(df["author"].isin(community_users))]
                community_counts = pd.DataFrame(filtered_df.groupby(on)["author"].size().reset_index(name="abs_count"))
                community_counts.sort_values(by="abs_count", inplace=True)
                community_counts["count"] = community_counts["abs_count"] / len(df[df[on] == community])
                community_counts.to_json(f"{os.path.join(folder, community)}.json")
                print(folder, community)

def get_all_users(on="group"):
    df_counts = pd.DataFrame(columns=["year", "month", on, "count"])
    communities = subreddits.group_names
    for root, dirs, files in os.walk(os.path.join(data_path, f"user_{on}_counts")):
        for file in files:
            year, month = os.path.splitext(file)[0].split("-")
            file_path = os.path.join(root, file)
            df = pd.DataFrame(pd.read_json(file_path))
            groups = df.groupby(on)
            for community in communities:
                try:
                    community_users = len(groups.get_group(community)["author"].unique())
                except KeyError:
                    community_users = 0
                df_counts = pd.concat([df_counts, pd.DataFrame(data=[(year, month, community, community_users)], columns=["year", "month", on, "count"])])
    df_counts.to_csv(os.path.join(data_path, f"{on}_abs_counts.csv"))



def create_graph_ml(on="group"):
    directories = []
    for i, (root, dirs, files) in enumerate(os.walk(os.path.join(data_path, f"{on}_overlaps"))):
        if i == 0:
            directories = dirs
        else:
            if os.path.exists(os.path.join(data_path, f"graphs_{on}/{directories[i - 1]}.graphml")):
                continue
            G = networkx.DiGraph()
            for file in files:
                df = pd.read_json(os.path.join(root, file))
                community = os.path.splitext(file)[0]
                for index, row in df.iterrows():
                    G.add_edge(community, row[on], weight=row["count"])
            networkx.readwrite.write_graphml(G, os.path.join(data_path, f"graphs_{on}/{directories[i - 1]}.graphml"))

def create_unidirectional_graph_ml(on="group", sim_type="jaccard", overwrite=False):
    directories = []
    os.makedirs(os.path.join(data_path, f"graphs_{on}_{sim_type}"), exist_ok=True)
    for i, (root, dirs, files) in enumerate(os.walk(os.path.join(data_path, f"{on}_overlaps"))):
        if i == 0:
            directories = dirs
        else:
            if not overwrite and os.path.exists(os.path.join(data_path, f"graphs_{on}_{sim_type}/{directories[i - 1]}.graphml")):
                continue
            G = networkx.Graph()
            comm_sizes = {}
            for file in files:
                df = pd.read_json(os.path.join(root, file))
                community = os.path.splitext(file)[0]
                if len(df[df["group"] == community]["abs_count"].values) != 1:
                    print("Ajjaj")
                    exit()
                comm_sizes[community] = df[df["group"] == community]["abs_count"].values[0]
            for file in files:
                df = pd.read_json(os.path.join(root, file))
                community = os.path.splitext(file)[0]
                for index, row in df.iterrows():
                    if comm_sizes[community] == 0 or comm_sizes[row[on]] == 0:
                        weight = 0
                    elif sim_type == "jaccard":
                        weight = row["abs_count"] / (comm_sizes[community] + comm_sizes[row[on]] - row["abs_count"])
                    elif sim_type == "overlap_coef":
                        weight = row["abs_count"] / min(comm_sizes[community], comm_sizes[row[on]])
                    print(directories[i - 1], community, row[on])
                    print(row["abs_count"], comm_sizes[community], comm_sizes[row[on]], row["abs_count"])
                    print(weight)
                    print("-----------")
                    if community != row[on]:
                        G.add_edge(community, row[on], weight=weight)
            networkx.readwrite.write_graphml(G, os.path.join(data_path, f"graphs_{on}_{sim_type}/{directories[i - 1]}.graphml"))


def create_network_trace(G, pos):
    edge_x = []
    edge_y = []
    edge_widths = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1])
        edge_y.extend([y0, y1])
        weight = edge[2].get('weight', 1)
        edge_widths.append(weight)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines')
    edge_trace['line'] = dict(width=0.1, color='#888')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                            marker=dict(showscale=True, colorscale='YlGnBu', color=[], size=10))

    return edge_trace, node_trace

def animate(on="group"):
    frames = []
    graphs = []
    all_nodes = set()
    all_edges = set()
    pos = None
    for root, dirs, files in os.walk(os.path.join(data_path, f"graphs_{on}")):
        for file in files:
            G = networkx.read_graphml(os.path.join(root, file))
            all_nodes.update(G.nodes())
            all_edges.update(G.edges())

    combined_graph = networkx.Graph()
    combined_graph.add_nodes_from(all_nodes)
    combined_graph.add_edges_from(all_edges)
    pos = networkx.spring_layout(combined_graph)  # Compute layout here

    for root, dirs, files in os.walk(os.path.join(data_path, f"graphs_{on}")):
        for file in files:
            G = networkx.read_graphml(os.path.join(root, file))
            # Use the consistent layout `pos` for every graph
            edge_trace, node_trace = create_network_trace(G, pos)
            graphs.append([edge_trace, node_trace])
            frames.append(go.Frame(data=[edge_trace, node_trace]))

    fig = go.Figure(
        data=graphs[0],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            updatemenus=[
                dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None])])]),
        frames=frames
    )

    fig.show()


def temporal_adjecency(sim_type=None, on="group"):
    graphs = []
    titles = []
    if sim_type == "directed" or sim_type is None:
        folder = f"graphs_{on}"
        frames_dir = os.path.join(data_path, f"{on}_adjacencies")
    else:
        folder = f"graphs_{on}_{sim_type}"
        frames_dir = os.path.join(data_path, f"{on}_adjacencies_{sim_type}")
    for root, dirs, files in os.walk(os.path.join(data_path, folder)):
        for file in files:
            G = networkx.read_graphml(os.path.join(root, file))
            graphs.append(G)
            titles.append(os.path.splitext(file)[-1])

    # Ensure this directory exists or adjust to a suitable location

    frame_files = []

    for i, G in enumerate(graphs):
        # Convert graph to a weighted adjacency matrix
        matrix = networkx.to_numpy_array(G, weight='weight')

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, cmap='viridis', annot=True, xticklabels=G.nodes(), yticklabels=G.nodes())
        plt.title(titles[i])
        frame_file = f'{frames_dir}/frame_{i}.png'
        plt.savefig(frame_file)
        plt.close()
        frame_files.append(frame_file)

    # Compile frames into a GIF
    gif_path = os.path.join(data_path, f"{frames_dir}/network_evolution.gif")
    with imageio.get_writer(gif_path, mode='I', duration=1) as writer:
        for filename in frame_files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def temporal_adjacency_plotly(sim_type=None, on="group"):
    graphs = []
    titles = []
    if sim_type == "directed" or sim_type is None:
        folder = f"graphs_{on}"
        frames_dir = os.path.join(data_path, f"{on}_adjacencies")
    else:
        folder = f"graphs_{on}_{sim_type}"
        frames_dir = os.path.join(data_path, f"{on}_adjacencies_{sim_type}")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    for root, dirs, files in os.walk(os.path.join(data_path, folder)):
        for file in files:
            G = networkx.read_graphml(os.path.join(root, file))
            graphs.append(G)
            titles.append(os.path.splitext(file)[0])
    frame_files = []

    for i, G in enumerate(graphs):
        # Convert graph to a weighted adjacency matrix
        zero_weight_edges = []
        for edge in G.edges():
            if edge[0] not in ["Democrat", "Republican"] and edge[1] not in ["Democrat", "Republican"]:
                zero_weight_edges.append(edge)
        G.remove_edges_from(zero_weight_edges)

        if on == "subreddit":
            nodes_to_remove = []
            for community in G.nodes():
                if community not in subreddits.subreddit_to_group:
                    nodes_to_remove.append(community)
            G.remove_nodes_from(nodes_to_remove)
            community_list = sorted(list(G.nodes()), key=lambda x: (subreddits.subreddit_to_group[x.lower()], x))
        else:
            community_list = sorted(list(G.nodes()))
            missing_communities = set([group for group in subreddits.subreddit_lists]) - set(community_list)
            for community in missing_communities:
                for group in ["Democrat", "Republican"]:
                    G.add_edge(community, group, weight=0)
                    G.add_edge(group, community, weight=0)
            community_list = sorted(list(G.nodes()))
        matrix = networkx.to_numpy_array(G, weight='weight', nodelist=community_list)
        text_matrix = np.where(matrix == 0, '', np.round(matrix, decimals=3).astype(str))
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=community_list,
            y=community_list,
            colorscale='Viridis',
            text=text_matrix,
            texttemplate="%{text}",  # Display the text as it is
            showlegend=False
        ))

        fig.update_layout(
            title=titles[i],
            xaxis=dict(title='Node'),
            yaxis=dict(title='Node'),
            width=2000,
            height=2000
        )

        frame_file = f'{frames_dir}/{titles[i]}.png'
        fig.write_image(frame_file)
        frame_files.append(frame_file)

    # Compile frames into a GIF
    gif_path = os.path.join(data_path, f"{frames_dir}/network_evolution_plotly.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
        for filename in frame_files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def temporal_adjacency_line_plot(sim_type=None, on="group"):
    graphs = []
    titles = []
    dates = []
    edge_weights = {}

    if sim_type == "directed" or sim_type is None:
        folder = f"graphs_{on}"
        frames_dir = os.path.join(data_path, f"{on}_adjacencies")
    else:
        folder = f"graphs_{on}_{sim_type}"
        frames_dir = os.path.join(data_path, f"{on}_adjacencies_{sim_type}")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    for root, dirs, files in os.walk(os.path.join(data_path, f"{folder}")):
        for file in files:
            G = networkx.read_graphml(os.path.join(root, file))
            zero_weight_edges = []
            for edge in G.edges():
                if edge[0] not in ["Democrat", "Republican"] and edge[1] not in ["Democrat", "Republican"]:
                    zero_weight_edges.append(edge)
            G.remove_edges_from(zero_weight_edges)
            graphs.append(G)
            titles.append(os.path.splitext(file)[0])

    for j, G in enumerate(graphs):
        for edge in G.edges(data=True):
            node1, node2, data = edge
            if (node1, node2) not in edge_weights:
                if (node2, node1) in edge_weights:
                    node2, node1 = node1, node2
                else:
                    edge_weights[(node1, node2)] = [0 for i in range(len(graphs))]
            edge_weights[(node1, node2)][j] = data['weight']

    edges_to_plot = list(edge_weights.keys())  # Modify this as needed

    fig = go.Figure()

    for edge in edges_to_plot:
        weights = edge_weights[edge]
        fig.add_trace(go.Scatter(
            x=titles,
            y=weights,
            mode='lines+markers',
            name=f'{edge[0]} to {edge[1]}'
        ))

    fig.update_layout(
        title='Edge Weight Evolution Over Time',
        xaxis=dict(title='Graph Index'),
        yaxis=dict(title='Weight'),
        width=1500,
        height=900
    )

    fig.show()
    fig.write_html(os.path.join(data_path, f"{frames_dir}/adjacency_evolution.html"))

def visualize_sizes():
    months = []
    sizes = collections.defaultdict(list)
    sizes["ChangeMyView"] = [0 for i in range(12)]
    overlap_folder = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data\group_overlaps"
    for root, dirs, files in os.walk(overlap_folder):
        for dir in sorted(dirs):
            months.append(dir)
            for file in os.listdir(os.path.join(root, dir)):
                community = file.split(".")[0]
                df = pd.read_json(os.path.join(root, dir, file))
                abs_count = df[df["group"] == community]["abs_count"].values[0]
                sizes[community].append(abs_count)
    pprint.pprint(sizes)

    months_dt = [datetime.strptime(m, "%Y-%m") for m in months]

    # Create Plotly figure
    fig = go.Figure()

    for community, counts in sizes.items():
        if community in ["Democrat", "Republican"]:
            fig.add_trace(go.Scatter(
                x=months_dt[:len(counts)],
                y=counts,
                mode='lines+markers',
                name=community
            ))

    fig.update_layout(
        title="Community Sizes Over Time",
        xaxis_title="Month",
        yaxis_title="Absolute Count",
        xaxis=dict(tickformat="%Y-%m"),
        hovermode='x unified'
    )

    fig.show()

visualize_sizes()

#if __name__ == '__main__':
    #get_user_activity_per_group()
    #get_overlaps(allow_self=True, overwrite=True)
    #create_unidirectional_graph_ml(overwrite=True)
    #temporal_adjacency_line_plot(sim_type="jaccard")
    #temporal_adjacency_plotly(sim_type="jaccard")
    #create_unidirectional_graph_ml(sim_type="overlap_coef", overwrite=True)
    #temporal_adjacency_line_plot(sim_type="overlap_coef")
    #temporal_adjacency_plotly(sim_type="overlap_coef")
