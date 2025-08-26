import ast
import collections
import os
import pprint
from datetime import datetime

import numpy as np
import pandas
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px
from plotly.subplots import make_subplots
import subreddits

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"


def permutation_test(series1, series2, num_permutations=10000):
    # Calculate the observed difference in means
    observed_diff = np.mean(series1) - np.mean(series2)

    # Combine the data
    combined = np.concatenate([series1, series2])

    # Perform permutations
    permuted_diffs = []
    for _ in range(num_permutations):
        # Shuffle the combined data
        np.random.shuffle(combined)
        # Split into two new samples
        new_series1 = combined[:len(series1)]
        new_series2 = combined[len(series1):]
        # Calculate the difference in means for the permuted samples
        permuted_diff = np.mean(new_series1) - np.mean(new_series2)
        permuted_diffs.append(permuted_diff)

    # Calculate the p-value
    permuted_diffs = np.array(permuted_diffs)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value


def funnel():

    stages = ["Thresholded", "Containing anomalic values", "Anomaly coinciding <br>with the event", "Bigram anomaly coinciding with <br>behavioral anomaly", "Statistically associated events"]
    df_mtl = pd.DataFrame(dict(number=[141, 139, 26, 6, 3], stage=stages))
    df_mtl['office'] = 'Republican'
    df_toronto = pd.DataFrame(dict(number=[157, 155, 22, 11, 5], stage=stages))
    df_toronto['office'] = 'Democrat'
    df = pd.concat([df_mtl, df_toronto], axis=0)
    fig = go.Figure()
    #fig = px.funnel(df, x='number', y='stage', color='office')
    #fig.show()
    data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"
    # Create the funnel chart
    fig.add_trace(go.Funnel(
        name='Democrat',
        orientation="h",
        y=stages,
        x=[157, 155, 22, 11, 5],
        textposition="auto",
        textinfo="value",
        textfont=dict(size=20)))

    fig.add_trace(go.Funnel(
        name='Republican',
        y=stages,
        x=[141, 139, 26, 6, 3],
        textposition="auto",
        textinfo="value",
        textfont=dict(size=20)))

    y_positions = list(range(len(stages)))

    # Add horizontal dashed lines
    shapes = []
    for i, pos in enumerate(y_positions[:-1]):  # Exclude the last position to avoid an extra line
        shapes.append(
            dict(
                type="line",
                x0=0,  # Start at the left of the plot
                x1=1,  # End at the right of the plot
                y0=pos / (len(stages) - 1),  # Normalized y position for the line
                y1=pos / (len(stages) - 1),  # Same y position for horizontal line
                line=dict(
                    color="blue",
                    width=1,
                    dash="dash"  # Dashed line style
                )
            )
        )

    # Adjust the layout to add more space at the top
    fig.update_layout(
        margin=dict(t=210),  # Increase top margin to 200 pixels to accommodate the arrows,
        paper_bgcolor="beige",
        plot_bgcolor="lightgray",
        yaxis=dict(
            tickfont=dict(size=16)  # Adjust font size and color for y-axis labels
        )
    )

    # Function to add three arrows for each annotation
    def add_triple_arrow_annotation(fig, x_center, y, text):
        offsets = [-0.04, 0.04, 0]  # Horizontal offsets for the arrows
        for offset in offsets:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=x_center + offset,  # Slightly offset each arrow horizontally
                y=y,  # Position the annotation text
                text=text if offset == 0 else "   ",  # Only show the text in the center arrow
                showarrow=True,
                arrowhead=3,
                arrowsize=2,  # Make the arrow wider and longer
                ax=0,
                ay=-70,  # Increase this value to move the arrows further down
                font=dict(size=32, color="gray"),
                align="center",
            )

    # Add the first annotation with three arrows
    add_triple_arrow_annotation(fig, x_center=0.5, y=1.22, text="28214 - All bigrams")

    # Add the second annotation with three arrows
    add_triple_arrow_annotation(fig, x_center=0.5, y=1.05, text="14768 - Bigrams mentioned on Reddit")

    # Show the figure
    fig.show()
    fig.update_layout(width=1500,  # Increase the width of the figure
        height=800,)
    fig.write_image(os.path.join(data_path, "funnel.png"))

def comparison(filename1=None, filename2=None,
               csv_file=True,
               sentiment=False,
               net_users=False,
               daily=False, date=None,
               value_col1="weights", value_col2=None,
               anomaly_col1="anomaly", anomaly_col2=None,
               anomalies=True,
               color_diff=False,
               anomaly_filename1=None, anomaly_filename2=None,
               title="", beh_type="",
               group1=None, group2=None):
    print(value_col1)
    if filename2 is None:
        filename2 = filename1
    if anomaly_filename1 is None:
        anomaly_filename1 = filename1
    if anomaly_filename2 is None:
        anomaly_filename2 = anomaly_filename1
    if value_col2 is None:
        value_col2 = value_col1
    if anomaly_col2 is None:
        anomaly_col2 = anomaly_col1
    if csv_file:
        filename1 = os.path.join(data_path, filename1)
        filename2 = os.path.join(data_path, filename2)
        df1 = pandas.read_csv(filename1, index_col=0, parse_dates=True)
        df2 = pandas.read_csv(filename2, index_col=0, parse_dates=True)
        if not daily:
            time_series_1 = df1[value_col1]
            time_series_2 = df2[value_col2]

            observed_diff, p_value = permutation_test(time_series_1, time_series_2, num_permutations=10000)
            print(observed_diff, p_value, abs(observed_diff)/min(time_series_1.mean(), time_series_2.mean()))
            if anomalies:
                anomaly_filename1 = os.path.join(data_path, anomaly_filename1)
                anomaly_filename2 = os.path.join(data_path, anomaly_filename2)
                bool_series_1 = pd.read_csv(anomaly_filename1, index_col=0, parse_dates=True)[anomaly_col1]
                bool_series_2 = pd.read_csv(anomaly_filename2, index_col=0, parse_dates=True)[anomaly_col2]
            else:
                bool_series_1 = []
                bool_series_2 = []
        else:
            time_series_1 = df1["value"]
            time_series_2 = df2["value"]
            bool_series_1 = df1.index == date
            bool_series_2 = df2.index == date
    elif sentiment:
        df1 = pandas.read_json(os.path.join(data_path, "sentiment_evolutions2.json"))
        df1.index = pd.DatetimeIndex(pd.date_range(pd.Timestamp("2012-01"), pd.Timestamp("2022-12"), freq="MS"))
        df1.drop(["year", "month"], inplace=True, axis="columns")
        time_series_1 = df1["mean_Democrat"]
        time_series_2 = df1["mean_Republican"]
        df2 = pandas.read_csv(os.path.join(data_path, "sentiment_evolutions2_anomalies.csv"), index_col=0, parse_dates=True)
        bool_series_1 = df2["mean_Democrat"]
        bool_series_2 = df2["mean_Republican"]
        #print(bool_series_1)
    elif net_users:
        with open(filename1, "rt") as f:
            time_series_1 = np.array(json.load(f)[group1])
        with open(filename2, "rt") as f:
            time_series_2 = np.array(list(reversed(json.load(f)[group1])))

    #print(time_series_1.mean(), time_series_2.mean(), time_series_1.mean() - time_series_2.mean())
    #print((time_series_2 - time_series_1).max())
    #print(time_series_1.std(), time_series_2.std())
    # Create the figure
    fig = go.Figure()

    # Calculate the differences
    if not net_users:
        differences = np.abs(time_series_1 - time_series_2)

        # Normalize differences for color mapping
        norm_differences = (differences - differences.min()) / (differences.max() - differences.min())
    else:
        differences = time_series_1 - time_series_2
        print()
        norm_differences = (differences) / (differences.max() - differences.min())
    try:
        x1 = time_series_1.index
    except:
        x1 = pd.DatetimeIndex(pd.date_range(pd.Timestamp("2012-01"), pd.Timestamp("2022-12"), freq="MS"))
    # Add the first time series
    if not net_users:
        trace_name_1 = 'Democrat' if not group1 else group1
        trace_name_2 ='Republican' if not group2 else group2
    else:
        trace_name_1 = f"New unique <br>{group1} users"
        trace_name_2 = f"Leaving unique <br>{group2} users"
    fig.add_trace(go.Scatter(
        x=x1,
        y=time_series_1,
        mode='lines',
        name=trace_name_1,
        line=dict(color='green')
    ))
    try:
        x2 = time_series_2.index
    except:
        x2 = pd.DatetimeIndex(pd.date_range(pd.Timestamp("2012-01"), pd.Timestamp("2022-12"), freq="MS"))
    # Add the second time series with area fill based on the difference
    fig.add_trace(go.Scatter(
        x=x2,
        y=time_series_2,
        mode='lines',
        name=trace_name_2,
        line=dict(color='red'),
        fill='tonexty' if color_diff else None,  # Fill to the previous y values
        fillcolor='rgba(255, 0, 0, 0.4)' if color_diff else None,  # Light red fill for the area
        showlegend=True
    ))

    # Calculate the color gradient based on the normalized differences
    if color_diff:
        print(norm_differences)
        for i in range(len(time_series_1) - 1):
            if not net_users:
                fillcolor = f'rgba({int(255 * (norm_differences[i]))}, {int(255 * (1 - norm_differences[i]))}, {64}, 0.4)'
            else:
                fillcolor = f'rgba({150 - int(255 * norm_differences[i]) if norm_differences[i] < 0 else 0}, ' \
                            f'{150 + int(255 * (norm_differences[i])) if norm_differences[i] > 0 else 0}, {64}, 0.4)'
                print(fillcolor)
            fig.add_trace(go.Scatter(
                x=[x1[i], x1[i + 1], x2[i + 1], x2[i]],
                y=[time_series_1[i], time_series_1[i + 1], time_series_2[i + 1], time_series_2[i]],
                fill='toself',
                fillcolor=fillcolor,
                line=dict(color='rgba(255, 255, 255, 0)'),  # Hide border lines
                showlegend=False,
                mode='lines'
            ))
        # Add anomalies for the first time series
    #print(bool_series_1)
    if anomalies:
        fig.add_trace(go.Scatter(
            x=x1[bool_series_1],
            y=time_series_1[bool_series_1],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Dem Anomaly'
        ))

        # Add anomalies for the second time series
        fig.add_trace(go.Scatter(
            x=x2[bool_series_2],
            y=time_series_2[bool_series_2],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Rep Anomaly'
        ))

    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=beh_type,
        legend_title="Time Series",
        width=1200,  # Increase the width of the figure
        height=800
    )

    # Show the figure
    fig.show()

    # Save the figure as an image
    fig.write_image(os.path.join(data_path, "net_users.png"))
    return fig

def get_all_bigram_events(group):
    df = pandas.read_csv(os.path.join(data_path, f"anomalic_bigrams_on_events_{group}.csv"), index_col=0,
                         parse_dates=True)
    event_bigrams = []
    for column in df.columns:
        if len(df[column][df[column]]) > 0:
            for x in df[column][df[column]].index.values:
                event_bigrams.append(((pandas.Timestamp(x).year, pandas.Timestamp(x).month), ast.literal_eval(column)))
    event_bigrams = sorted(event_bigrams, key=lambda x: (x[0][0], x[0][1]))
    pprint.pprint(event_bigrams)

    with open(os.path.join(data_path, "events_bigrammed.json"), "rt") as f:
        events_bigrammed = json.load(f)
    with open(os.path.join(data_path, "events_bigrammed_years.json"), "rt") as f:
        events_bigrammed_years = json.load(f)

    event_dates = []

    for (date, bigram) in event_bigrams:
        found = False
        month = events_bigrammed[str(date[1])]
        for day in month:
            for i, bigram_list in enumerate(month[str(day)]):
                for event_bigram in bigram_list:
                    if event_bigram[0] == bigram[0] and event_bigram[1] == bigram[1]:
                        # print(date[0], events_bigrammed_years[str(date[1])][day][i])
                        if date[0] == events_bigrammed_years[str(date[1])][day][i] and not found:
                            found = True
                            event_dates.append((f"{date[0]}-{date[1]}-{int(day)}"))
                            print(bigram)

        if not found:
            print(date, bigram, "Ajjaj")
    print(event_dates)
    print(len(set(event_dates)))
    print(sorted(list(set(event_dates)),
                 key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2]))
                 ))


def get_behavior_anomaly_bigram_events(group):
    with open(os.path.join(data_path, f"anomalic_events_bigrams_behaviors_merged_{group}.json")) as f:
        json_file = json.load(f)
    event_bigrams = []
    for beh_type in json_file:
        for beh_val in json_file[beh_type]:
            for bigram, date in json_file[beh_type][beh_val].items():
                event_bigrams.append(((date[0][0], date[0][1]), ast.literal_eval(bigram)))

    event_bigrams = sorted(event_bigrams, key=lambda x: (x[0][0], x[0][1]))
    pprint.pprint(event_bigrams)

    with open(os.path.join(data_path, "events_bigrammed.json"), "rt") as f:
        events_bigrammed = json.load(f)
    with open(os.path.join(data_path, "events_bigrammed_years.json"), "rt") as f:
        events_bigrammed_years = json.load(f)

    event_dates = []

    for (date, bigram) in event_bigrams:
        found = False
        month = events_bigrammed[str(date[1])]
        for day in month:
            for i, bigram_list in enumerate(month[str(day)]):
                for event_bigram in bigram_list:
                    if event_bigram[0] == bigram[0] and event_bigram[1] == bigram[1]:
                        # print(date[0], events_bigrammed_years[str(date[1])][day][i])
                        if date[0] == events_bigrammed_years[str(date[1])][day][i] and not found:
                            found = True
                            event_dates.append((f"{date[0]}-{date[1]}-{int(day)}"))
                            print(bigram)

        if not found:
            print(date, bigram, "Ajjaj")
    print(event_dates)
    print(len(set(event_dates)))
    print(sorted(list(set(event_dates)),
                 key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2]))
                 ))
    return event_dates


def get_anomalic_bigram_events(group):
    event_dates = get_behavior_anomaly_bigram_events(group)
    if group == "Democrat":
        event_names = [
            "Tammy Baldwin, First Openly Gay Woman Senator",
            "Sandy Hook School Shooting",
            "Murder of Michael Brown",
            "Hillary Clinton Becomes Dem. Nominee",
            "2016 US Elections",
            "Women's March",
            "Chemical Weapons Attack In Syria",
            "Hurricane Maria in Puerto Rico",
            "Las Vegas Mass Shooting",
            "El Paso Mass Shooting",
            "Murder of George Floyd",
            "Moskva Submarine Sunk By Ukraine",
            "Buffalo Mass Shooting"
        ]
        colors = [
            ["lightblue"],
            ["orange"],
            ["lightblue"],
            ["lightblue", "blue", "cyan"],
            ["lightgreen", "lightblue", "cyan", "yellow"],
            ["lightblue"],
            ["blue"],
            ["lightgreen"],
            ["lightblue"],
            ["lightgreen"],
            ["lightblue", "yellow"],
            ["red"],
            ["yellow"]
        ]
    else:
        event_names = [
            "Sandy Hook School Shooting",
            "2016 US Elections",
            "Las Vegas Mass Shooting",
            "El Paso Mass Shooting",
            "Donald Trump Declares COVID-19 National Emergency",
            "Bernie Sanders Ends Presidential Campaign",
        ]
        colors = [
            ["orange"],
            ["lightgreen", "lightblue", "cyan", "yellow"],
            ["lightblue", "blue"],
            ["lightgreen"],
             ["lightblue", "cyan", "yellow"],
             ["yellow"],
             ]
    print(len(event_names), len(list(set(event_dates))))
    events = {
        'Date': sorted(list(set(event_dates)),
                 key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2]))
                 ),
        'Event': event_names
    }

    # Convert to DataFrame
    df = pandas.DataFrame(events)
    df['Date'] = pandas.to_datetime(df['Date'])

    grouped = df.groupby('Date').agg({'Event': '\n'.join}).reset_index()

    # Calculate time differences in days from the start date
    grouped['Days'] = (grouped['Date'] - pd.Timestamp(year=2012, month=1, day=1)).dt.days

    # Normalize Days to fit the timeline width
    grouped['Position'] = grouped['Days'] / (pd.Timestamp(year=2023, month=1, day=1) - pd.Timestamp(year=2012, month=1, day=1)).days

    # Set Seaborn style for a more polished look
    sns.set(style="whitegrid")

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(13, 5))  # Increased figure size for more content

    # Central horizontal line with enhanced styling
    ax.axhline(0.5, color='gray', linewidth=1.5, linestyle='-', alpha=0.7)

    #ax.text(-0.05, 0.52, "2012-01", ha='right', va='center', fontsize=10, color='black')

    # Add text at the right end
    #ax.text(1.05, 0.52, "2022-12", ha='left', va='center', fontsize=10, color='black')
    start_date = "2012-01"
    end_date = "2022-12"
    dates = pd.date_range(start=start_date, end=end_date, freq='3MS')

    # Calculate the position for each date as a fraction of the total length
    total_months = (dates[-1].year - dates[0].year) * 12 + dates[-1].month - dates[0].month

    for date in dates:
        # Calculate the position of the date as a fraction of the total time span
        months_passed = (date.year - dates[0].year) * 12 + date.month - dates[0].month
        position = (date - pd.Timestamp(year=2012, month=1, day=1)).days / (pd.Timestamp(year=2023, month=1, day=1) - pd.Timestamp(year=2012, month=1, day=1)).days

        # Add a small vertical mark at each position
        ax.plot([position, position], [0.49, 0.51], color='black', linewidth=1)

        # Add the date label
        ax.text(position, 0.48, date.strftime('%Y-%m'), ha='center', va='top', fontsize=8, rotation=90)

    # Add an arrow at the right end of the line
    ax.annotate('', xy=(1.05, 0.5), xytext=(0.99, 0.5),
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.5))

    # Define a color palette
    palette = sns.color_palette("pastel", len(grouped))

    # Plot each event
    for i, (date, position, events) in enumerate(zip(grouped['Date'], grouped['Position'], grouped['Event'])):
        alt_position = 0.5 + (-1) ** i * (0.2 if (i % 6) < 2 else 0.15 if (i % 6) < 4 else 0.25)  # Closer bubbles
        #box_color = palette[i % len(palette)]  # Use color from palette
        box_colors = colors[i]
        # Create a gradient box if there are multiple colors
        # Combine the date with the events
        label = f"{date.strftime('%b %d, %Y')}\n{events}"

        # Draw a connecting line with a slight shadow effect
        ax.plot([position, position], [0.5, alt_position], color='darkblue', linestyle='-', linewidth=1.5, alpha=0.8)
        for j, coljr in reversed([(a, b) for a, b in enumerate(box_colors)]):
            print(len(label))
            text = "".join([" " for i in range(len(label)//len(box_colors))])
            if len(box_colors) % 2 == 1:
                ax.text(position + len(text) * (j - len(box_colors)//2) * 0.005, alt_position , text, ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.4" if j == 0 or j == len(box_colors) - 1 else "square, pad=0.4", edgecolor="none", facecolor=coljr, alpha=0.9))
            else:
                ax.text(position + len(text) * (j - len(box_colors) // 2 + 0.5) * 0.005, alt_position, text, ha='center',
                        va='center', fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.4" if j == 0 or j == len(box_colors) - 1 else "square, pad=0.4", edgecolor="none", facecolor=coljr, alpha=0.9))

        # Add the actual text on top
        ax.text(position, alt_position, label, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", edgecolor="none", facecolor="none"))
        # Add event boxes with subtle shadow and padding
        #ax.text(position, alt_position, label, ha='center', va='center', fontsize=9,
        #        bbox=dict(boxstyle="round,pad=0.4", edgecolor="none", facecolor=box_colors[0] if len(box_colors) == 1 else None, alpha=0.9))

    # Set background color
    ax.set_facecolor('#f0f0f0')
    ax.grid(False)

    # Adjust the plot limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.2, 0.8)
    ax.axis('off')  # Hide the axes


    # Title with increased font size
    plt.title('Events with Coinciding Anomalic Bigrams and Anomalic Behavioral Values', fontsize=18, fontweight='bold', color='darkblue')
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"event_anomaly_timeline_{group}.png"))
    plt.show()

def anomaly_counts():
    # Create the figure and axis with a slightly different Seaborn style
    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 6))

    series_a = pd.read_csv(os.path.join(data_path, f"anomaly_counts_Democrat.csv"), index_col=0, parse_dates=True)["cnts"]
    series_b = pd.read_csv(os.path.join(data_path, f"anomaly_counts_Republican.csv"), index_col=0, parse_dates=True)["cnts"]
    print(series_a.sum(), series_b.sum())
    time_points = series_a.index

    # Enhanced color palette
    palette = sns.color_palette(["#4C72B0", "#C44E52"])

    # Step plot for Series A with enhanced markers and line styles
    plt.step(time_points, series_a, where='mid', label='Democrat',
             linestyle='-', marker='o', color=palette[0], linewidth=2.5, markersize=10, markerfacecolor='white',
             markeredgewidth=2)

    # Step plot for Series B with enhanced markers and line styles
    plt.step(time_points, series_b, where='mid', label='Republican',
             linestyle='--', marker='s', color=palette[1], linewidth=2.5, markersize=10, markerfacecolor='white',
             markeredgewidth=2)

    # Adding titles and labels with enhanced fonts
    plt.title('Number of anomalic values in each month', fontsize=20, fontweight='bold')
    plt.xlabel('Time', fontsize=16, labelpad=10)
    plt.ylabel('Count', fontsize=16)

    # Adding legend with enhanced style
    plt.legend(title='Series', title_fontsize='14', fontsize='12', loc='upper right')

    # Show plot with grid and adjusted aesthetics
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    sns.despine(trim=True)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, fontsize=12)

    # Adjust layout to prevent cropping of labels
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, "anomaly_counts.png"))
    plt.show()

#comparison(os.path.join("meta", "graphs", "jaccard", f"new_Republican - Conspiracy - 2017-10-01.csv"),
#           os.path.join("meta", "graphs", "jaccard", f"old_Republican - Conspiracy - 2017-10-01.csv"),
#           daily=True,
#           date=pd.Timestamp(year=2017, month=10, day=1))
#comparison(f"anomaly_counts_Democrat.csv",
#           f"anomaly_counts_Republican.csv",
#           value_col="cnts",
#           anomalies=False)
#comparison(sentiment=True, csv_file=False)
#funnel()

def att_flow_anomalies():
    rep = "Republican"
    dem = "Democrat"
    figures = []
    for comm in subreddits.subreddit_lists:
        if comm == "Democrat":
            continue
            fig = comparison(filename1="attention_flow.csv", anomaly_filename1="attention_flow_anomalies.csv",
                             value_col1=f"{dem} - {rep}",
                             value_col2=f"{rep} - {dem}",
                             anomaly_col1=f"{dem} - {rep}",
                             anomaly_col2=f"{rep} - {dem}",
                             title=f"Between Dem and Rep")
            figures.append(fig)
        elif comm == "Republican":
            continue
        else:
            fig = comparison(filename1="attention_flow.csv", anomaly_filename1="attention_flow_anomalies.csv",
                    value_col1=f"{comm} - {dem}",
                    value_col2=f"{comm} - {rep}",
                    anomaly_col1=f"{comm} - {dem}",
                    anomaly_col2=f"{comm} - {rep}",
                    title=f"Incoming from {comm}")
            figures.append(fig)
            fig = comparison(filename1="attention_flow.csv", anomaly_filename1="attention_flow_anomalies.csv",
                             value_col1=f"{dem} - {comm}",
                             value_col2=f"{rep} - {comm}",
                             anomaly_col1=f"{dem} - {comm}",
                             anomaly_col2=f"{rep} - {comm}",
                             title=f"Outgoing to {comm}",
                             beh_type="Attention flow")
            figures.append(fig)
    print(len(figures))
    rows = 8
    cols = 2
    subplot_titles = [sub_fig.layout.title.text if 'title' in sub_fig.layout and sub_fig.layout.title.text else f"Figure {i+1}" for i, sub_fig in enumerate(figures)]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, vertical_spacing=0.05)
    for i, plot in enumerate(figures):
        row = (i // cols) + 1
        col = (i % cols) + 1
        for trace in plot.data:
            fig.add_trace(trace, row=row, col=col)
        if 'title' in plot.layout:
            fig.layout.annotations[i].update(text=plot.layout.title.text)

    # Update layout if necessary
    fig.update_layout(
        height=1600,
        width=1000,
        showlegend=False,
        margin=dict(l=5, r=5, b=5)  # Set all margins to 0
    )

    # Show the combined grid figure
    fig.show()
    fig.write_image(os.path.join(data_path, "attention_flow_comparisons.png"))


def calendar_heatmap(dates, counts, title):
    """Render a calendar-style heat‑map (years × months)."""
    df = pd.DataFrame({"date": dates, "count": counts})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    heat = (
        df.pivot_table(index="year", columns="month", values="count", aggfunc="sum")
        .reindex(sorted(df["year"].unique(), reverse=True))
        .reindex(range(1, 13), axis=1)  # ensure all months present
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat)
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        [i + 1 for i in range(12)]
    )
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    plt.colorbar(im, ax=ax, label="New users / month")
    ax.set_title(title)
    fig.tight_layout()
    return fig




def visualize_new_users():
    months = []
    groups = ["Democrat", "Republican"]
    users_path = os.path.join(data_path, "user_group_counts_all")
    first_month_df = pd.read_json(os.path.join(users_path, "2012-01.json"))
    existing_users = {
        group: set(first_month_df[first_month_df["group"] == group]["author"].values) for group in groups
    }
    new_counts = collections.defaultdict(list)
    ratios = collections.defaultdict(list)
    for root, dirs, files in os.walk(users_path):
        for file in sorted(files):
            months.append(file.split(".")[0])
            df = pd.read_json(os.path.join(users_path, file))
            for group in groups:
                df_group = df[df["group"] == group]
                df_group_users = set(df_group["author"].values)
                new_counts[group].append(len(df_group_users - existing_users[group]))
                existing_users[group].update(df_group_users)
                print(file, len(existing_users[group]), new_counts[group][-1], len(df_group_users))
                ratios[group].append(new_counts[group][-1] / len(df_group_users))

    pprint.pprint(new_counts, indent=4)
    for group in groups:
        print(min(ratios[group][1:]))
        print(months[ratios[group].index(min(ratios[group][1:]))])
        print(max(ratios[group][1:]))
        print(months[ratios[group].index(max(ratios[group][1:]))])
        print(sum(ratios[group])/len(ratios[group]))

    months_dt = [datetime.strptime(m, "%Y-%m") for m in months]

    # Create Plotly figure
    fig = go.Figure()

    for community, counts in new_counts.items():
        if community in ["Democrat", "Republican"]:
            fig.add_trace(go.Scatter(
                x=months_dt[:len(counts)],
                y=counts,
                mode='lines+markers',
                name=community
            ))

    fig.update_layout(
        title="Number of new users",
        xaxis_title="Month",
        yaxis_title="Absolute Count",
        xaxis=dict(tickformat="%Y-%m"),
        hovermode='x unified'
    )

    fig.write_image("new_users_overtime.png")

    fig.show()

    # --------- One chart per group (separate figures, no subplots) ---------
    fig_dem = calendar_heatmap(months_dt, new_counts["Democrat"],
                               "First‑time Democrat Users")
    fig_dem.savefig("new_users_heatmap_Democrat.png")
    fig_dem.show()
    fig_rep = calendar_heatmap(months_dt, new_counts["Republican"],
                               "First‑time Republican Users")
    fig_rep.savefig("new_users_heatmap_Republican.png")
    fig_rep.show()

    with open("new_users.json", "wt") as f:
        json.dump(new_counts, f)


def visualize_leaving_users():

    with open(os.path.join(data_path, "all_users.json"), "rt") as f:
        all_users = set(json.load(f))
    months = []
    groups = ["Democrat", "Republican"]
    all_users = {group: all_users for group in groups}
    users_path = os.path.join(data_path, "user_group_counts_all")
    leaver_counts = collections.defaultdict(list)
    ratios = collections.defaultdict(list)
    for root, dirs, files in os.walk(users_path):
        for file in reversed(sorted(files)):
            months.append(file.split(".")[0])
            df = pd.read_json(os.path.join(users_path, file))
            for group in groups:
                df_group = df[df["group"] == group]
                df_group_users = set(df_group["author"].values)
                month_leavers = all_users[group].intersection(df_group_users)
                all_users[group] = all_users[group] - month_leavers
                leaver_counts[group].append(len(month_leavers))
                ratios[group].append(len(month_leavers)/len(df_group_users))
                print(file, len(all_users[group]), leaver_counts[group][-1], len(df_group_users))

    pprint.pprint(leaver_counts, indent=4)
    for group in groups:
        print(min(ratios[group][1:]))
        print(months[ratios[group].index(min(ratios[group][1:]))])
        print(max(ratios[group][1:]))
        print(months[ratios[group].index(max(ratios[group][1:]))])
        print(sum(ratios[group]) / len(ratios[group]))

    months_dt = [datetime.strptime(m, "%Y-%m") for m in months]

    # Create Plotly figure
    fig = go.Figure()

    for community, counts in leaver_counts.items():
        if community in ["Democrat", "Republican"]:
            fig.add_trace(go.Scatter(
                x=months_dt[:len(counts)],
                y=counts,
                mode='lines+markers',
                name=community
            ))

    fig.update_layout(
        title="Number of leaving users",
        xaxis_title="Month",
        yaxis_title="Absolute Count",
        xaxis=dict(tickformat="%Y-%m"),
        hovermode='x unified'
    )
    fig.write_image("leaver_users_overtime.png")
    fig.show()



    # --------- One chart per group (separate figures, no subplots) ---------
    fig_dem = calendar_heatmap(months_dt, leaver_counts["Democrat"],
                               "Leaving Democrat Users")
    fig_dem.savefig("leaving_users_heatmap_Democrat.png")
    fig_dem.show()
    fig_rep = calendar_heatmap(months_dt, leaver_counts["Republican"],
                               "Leaving Republican Users")
    fig_rep.savefig("leaver_users_heatmap_Republican.png")
    fig_rep.show()

    with open("leaver_users.json", "wt") as f:
        json.dump(leaver_counts, f)

def get_net_user_growth(group):
    with open("leaver_users.json", "rt") as f:
        leaver_users = json.load(f)[group]

    leaver_users = list(reversed(leaver_users))

    with open("new_users.json", "rt") as f:
        new_users = json.load(f)[group]

    net_users = np.array(new_users) - np.array(leaver_users)
    net_users = net_users[:-4]

    users_path = os.path.join(data_path, "user_group_counts_all")
    months = []
    for root, dirs, files in os.walk(users_path):
        for file in (sorted(files)):
            months.append(file.split(".")[0])

    months_dt = [datetime.strptime(m, "%Y-%m") for m in months]

    # Create Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=months_dt[:len(net_users)],
        y=net_users,
        mode='lines+markers',
        name=group
    ))

    fig.update_layout(
        title="Net users per month",
        xaxis_title="Month",
        yaxis_title="Absolute Count",
        xaxis=dict(tickformat="%Y-%m"),
        hovermode='x unified'
    )
    fig.write_image("net_users_overtime.png")
    fig.show()

comparison("new_users.json", "leaver_users.json", csv_file=False, net_users=True, anomalies=False, color_diff=True, group1="Republican", group2="Republican", beh_type="User Count")
