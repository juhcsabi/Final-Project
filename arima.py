import ast
import json
import os.path
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import pmdarima as pm
import plotly.graph_objects as go

data_path = r"C:\Users\csabs\OneDrive - University of Twente\UT\Thesis\data"


def arima_analysis(filename=None,
                   column_to_check="weights",
                   overwrite=False,
                   non_df=False,
                   data_column=None,
                   ret_residuals=False,
                   include_if=True,
                   group=None):
    if not non_df:
        data = pd.read_csv(filename, index_col='dates', parse_dates=True)
        if "anomaly" in data.columns and not overwrite:
            return data["anomaly"], data["anomaly"]
        data_series = data[column_to_check].dropna()
    else:
        data_series = data_column

    if column_to_check == "weights":
        model = pm.auto_arima(data_series,
                          start_p=0, start_q=0,
                          max_p=2, max_q=2,
                          max_d=2,
                          seasonal=True,
                          m=12,
                          stepwise=False,
                          trace=False,
                          maxiter=1000)
    elif column_to_check == "detrended":
        model = pm.auto_arima(data_series,
                              start_p=0, start_q=0,
                              max_p=2, max_q=2,
                              seasonal=False,
                              stepwise=True,
                              trace=False,
                              maxiter=500)

    print(model.summary())
    print(model.order, model.maparams())

    arima_result = SARIMAX(data_series,
                           order=model.order,
                           seasonal_order=model.seasonal_order,
                           freq=data_series.index.inferred_freq,
                           dates=data_series.index).fit(method='powell')

    residuals = arima_result.resid
    residuals.iloc[0] = 0

    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)

    z_scores = zscore(residuals)

    z_threshold = 3
    anomalies_z = residuals[np.abs(z_scores) > z_threshold]
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    anomalies_if = iso_forest.fit_predict(residuals.values.reshape(-1, 1))
    anomalies_if_mask = anomalies_if == -1


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=residuals.index, y=residuals, mode='lines', name='Residuals'))
    fig.add_trace(go.Scatter(x=anomalies_z.index, y=anomalies_z, mode='markers',
                             marker=dict(color='red'), name='Anomalies (Z-Score)'))
    if include_if:
        fig.add_trace(go.Scatter(x=residuals.index[anomalies_if_mask], y=residuals[anomalies_if_mask], mode='markers',
                             marker=dict(color='green'), name='Anomalies (IF)'))


    fig.add_hline(y=3 * std_residuals, line=dict(color='blue', dash='dash'),
                  annotation_text='Threshold (Z-Score)', annotation_position='bottom right')
    fig.add_hline(y=-3 * std_residuals, line=dict(color='blue', dash='dash'))

    fig.update_layout(
        title=os.path.splitext(filename)[0].split("\\")[-1],
        xaxis_title='Index',
        yaxis_title='Value',
        legend=dict(x=0, y=-0.2),
        width=1200,
        height=600
    )

    print(os.path.join(data_path, "arima_res", os.path.splitext(filename)[0].split("\\")[-1] + ".html"))
    path_parts = os.path.splitext(filename)[0].split('\\')
    if group is None:
        fig.write_html(os.path.join(data_path, "arima_res", f"{path_parts[-2]}_{path_parts[-1]}.html"))
        fig.write_image(os.path.join(data_path, "arima_res", f"{path_parts[-2]}_{path_parts[-1]}.png"))
    else:
        fig.write_html(os.path.join(data_path, "arima_res", os.path.splitext(filename)[0].split("\\")[-1] + f"_{group}.html"))
        fig.write_image(os.path.join(data_path, "arima_res", os.path.splitext(filename)[0].split("\\")[-1] + f"_{group}.png"))

    print(anomalies_z)
    #print(data_series[anomalies_if_mask])
    print("----------")
    if ret_residuals:
        return residuals
    else:
        return anomalies_z, data_series[anomalies_if_mask]

def graph_arima(folder="tss_community_feedback", overwrite=True):
    for root, dirs, files in os.walk(os.path.join(data_path, folder)):
        for file in files:
            if ".html" in file:
                continue
            filename = os.path.join(root, file)
            print(os.path.splitext(file)[0])
            column_to_check = "weights"
            anomalies_z, anomalies_if = arima_analysis(filename, column_to_check, overwrite)
            df = pd.read_csv(filename, index_col='dates', parse_dates=True)
            df["anomaly"] = False

            if column_to_check == "detrended":
                df = df.dropna()
            fig = go.Figure()

            df["anomaly"][anomalies_z.index] = True
            fig.add_trace(go.Scatter(x=df.index, y=df["weights"], mode='lines', name='Residuals'))

            fig.add_trace(go.Scatter(x=anomalies_z.index, y=df.loc[anomalies_z.index]["weights"], mode='markers',
                                     marker=dict(color='blue'), name='Anomalies (Z-Score)'))

            #fig.add_trace(
                #go.Scatter(x=anomalies_if.index, y=df.loc[anomalies_if.index]["weights"], mode='markers',
                           #marker=dict(color='green'), name='Anomalies (IF)'))

            fig.update_layout(
                title=os.path.splitext(filename)[0].split("\\")[-1],
                xaxis_title='Index',
                yaxis_title='Value',
                legend=dict(x=0, y=-0.2),
                width=1200,
                height=600
            )

            #fig.show()
            fig.write_html(
                os.path.join(data_path, "arima", f"{folder} - {os.path.splitext(file)[0]}.html"))
            fig.write_image(
                os.path.join(data_path, "arima", f"{folder} - {os.path.splitext(file)[0]}.png")
            )
            if len(anomalies_z) > 0:
                print(df["anomaly"].values)
            df.to_csv(filename)


def convert_type(x):
    if str(x) == '0.0':
        return False
    if str(x).lower() == "true":
        return True
    print(x, '!!!!!!!!!')
    exit()

def arima_columns(filename="bigrams_reddit_relevant_thresholded.csv", bigrams=True, group=None):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_path, filename), index_col=0, parse_dates=True).fillna(0)
    elif filename.endswith(".json"):
        df = pd.read_json(os.path.join(data_path, filename))
        df.index = pd.DatetimeIndex(pd.date_range(pd.Timestamp("2012-01"), pd.Timestamp("2022-12"), freq="MS"))
        df.drop(["year", "month"], inplace=True, axis="columns")
    df_anomalies = df.copy(deep=True)
    #df_anomalies_existing = pd.read_csv(os.path.join(data_path, "bigrams_reddit_relevant_thresholded_anomalies2.csv"), index_col=0, parse_dates=True)
    for col in df_anomalies.columns:
        df_anomalies[col] = df_anomalies[col].astype("bool")
        df_anomalies[col].values[:] = False
    #for col in df_anomalies_existing.columns:
        #df_anomalies_existing[col] = df_anomalies_existing[col].astype(bool)
    with open(os.path.join(data_path, "anomalic_events_bigrams_behaviors_merged.json"), "rt") as f:
        json_file = json.load(f)
    anomalic_bigrams = set()
    for behavior_cat, behavior_cat_dict in json_file.items():
        for behavior, bigrams_dict in behavior_cat_dict.items():
            for bigram in bigrams_dict:
                anomalic_bigrams.add(bigram)
    for column in df.columns:
        if bigrams:
            (token1, token2) = ast.literal_eval(column)
            if column not in anomalic_bigrams:
                print(column, "not in anomalic bigrams")
            anomalies_z, anomalies_if = arima_analysis(non_df=True, data_column=df[column],
                                                       filename=f"{token1}-{token2}", overwrite=True, group=group)
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name='Time Series'))

            fig.add_trace(go.Scatter(x=anomalies_z.index, y=df.loc[anomalies_z.index][column], mode='markers',
                                     marker=dict(color='red'), name='Anomalies (Z-Score)'))

            fig.update_layout(
                title=os.path.splitext(filename)[0].split("\\")[-1],
                xaxis_title='Index',
                yaxis_title='Value',
                legend=dict(x=0, y=-0.2),
                width=1200,
                height=600
            )

            if group is None:
                fig.write_html(
                    os.path.join(data_path, "arima", "bigrams", f"{token1} - {token2} - {os.path.splitext(filename)[0]}.html"))
                fig.write_image(
                    os.path.join(data_path, "arima", "bigrams", f"{token1} - {token2} - {os.path.splitext(filename)[0]}.png")
                )
            else:
                fig.write_html(
                    os.path.join(data_path, "arima", "bigrams", group,
                                 f"{token1} - {token2} - {os.path.splitext(filename)[0]}.html"))
                fig.write_image(
                    os.path.join(data_path, "arima", "bigrams", group,
                                 f"{token1} - {token2} - {os.path.splitext(filename)[0]}.png")
                )
        else:
            print(column)
            df[column].fillna(df[column].mean(), inplace=True)
            anomalies_z, anomalies_if = arima_analysis(non_df=True, data_column=df[column],
                                                       filename=column, overwrite=True, include_if=False)
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name='Time Series'))

            fig.add_trace(go.Scatter(x=anomalies_z.index, y=df.loc[anomalies_z.index][column], mode='markers',
                                     marker=dict(color='red'), name='Anomalies (Z-Score)'))

            fig.update_layout(
                title=os.path.splitext(filename)[0].split("\\")[-1],
                xaxis_title='Index',
                yaxis_title='Value',
                legend=dict(x=0, y=-0.2),
                width=1200,
                height=600
            )

            # fig.show()
            fig.write_html(
                os.path.join(data_path, "arima", f"{column} - {os.path.splitext(filename)[0]}.html"))
            fig.write_image(
                os.path.join(data_path, "arima", f"{column} - {os.path.splitext(filename)[0]}.png")
            )

        df_anomalies[column][anomalies_z.index] = True
        df_anomalies.to_csv(os.path.join(data_path, f"{os.path.splitext(filename)[0]}_{group}_anomalies.csv"))

if __name__ == '__main__':
    #graph_arima()
    #graph_arima("tss_toxicities", overwrite=True)
    #graph_arima("time_series")
    #graph_arima("time_series_jaccard")
    #graph_arima("time_series_overlap_coef")
    #arima_columns(filename="bigram_counts_Democrat_thresholded.csv", group="Democrat")
    #arima_columns(filename="bigram_counts_Republican_thresholded.csv", group="Republican")
    #arima_columns("sentiment_evolutions2.json", bigrams=False)
    #arima_columns("attention_flow.csv", bigrams=False)
    arima_analysis(
        filename=os.path.join(data_path, "tss_community_feedback", "Democrat_Avg. comments per author.csv"),
        overwrite=True
    )
    arima_analysis(
        filename=os.path.join(data_path, "tss_community_feedback", "Republican_Avg. comments per author.csv"),
        overwrite=True
    )
    arima_analysis(
        filename=os.path.join(data_path, "tss_community_feedback", "Republican_Number of authors.csv"),
        overwrite=True
    )




