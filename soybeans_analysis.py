# %%
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import plotly.express as px
from matplotlib import colors
from os import listdir
from os.path import isfile, join
import numpy as np
import streamlit as st

soy_data = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soytrade_dataset.csv')
flour_data = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/flour_dataset.csv')

option = st.selectbox(
    'What product you want to analyse?',
    ('Soy beans', 'Flour',))

st.write('You selected:', option)

def display_analysis(product = 'Soy beans'):
    st.title(f'{product} Trade Analysis')
    if product == 'Flour':
        data = flour_data
    else:
        data = soy_data
    """
    To start with the analysis, we can take a look at the data. To keep things simpler I had preprocessed the data and build a dataset containing complementary 
    information about the countries. Besides the value traded, the size of each country, the GDP information, exchagenge rate, price index were also included.

    An important thing to notice is that the trade value is also presented in log scale. This is done to avoid the skewness of the data.
    """
    st.write(data)

    st.header("Trade x Area relationship")

    """
    The initial analysis is to check the relationship between the trade value and the area of the countries. 
    To investigate the different distribution of the trade value thought the years, a slider is provided to select the year of interest and a radio selector to select the country of interest.

    """

    year = st.slider(
        "Select the year you want to insepct",
        min_value=1997,
        max_value=2020,
        value=2017,
        key=f"year{product}")

    country = st.radio(
        "Which country do you want to inspect?",
        ('BRA', 'HUN', 'USA', 'CHN', 'ARG', 'CAN', 'FRA', 'IDN', 'IND', 'ITA', 'JPN', 'MEX', 'NLD', 'POL', 'RUS', 'TUR', 'UKR', 'VNM', 'ZAF'),
        key=f"country{product}",
        horizontal=True)

    st.subheader(f"You selected {country} and {year}")

    """
    Here the trade data is presented by the log of the trade value and the area of the partner countries.
    """
    #### create space for the replacement of the chart
    chart_placeholder = st.empty()
    chart_placeholder.pyplot()
    chart = sns.jointplot(x="ln_trade_value", y="Target Area", data=data[ (data['Source'] == country) & (data['Year'] == year)],
                    kind="reg", truncate=False,
                    color="m", height=7)
    chart.set_axis_labels('Trade Value (Ln)', 'Target Area', fontsize=16)
    chart_placeholder.pyplot(chart, use_container_width=True)

    """
    In order to inspect the 'outliers' of the previous chart, one can visualize the data in an interactive way.
    """
    trade_analysis = data[ (data['Source'] == country) & (data['Year'] == year)]

    fig = px.scatter(trade_analysis, y="Target Area", x="trade_value", text="Target", log_x=True, size_max=100, color="trade_value", trendline="ols")
    fig.update_traces(textposition='top center')
    fig.update_layout(title_text='Country Partner Area', title_x=0.5)
    # Plot!
    st.plotly_chart(fig, use_container_width=True)

    st.header("Countrys participation on trade")
    """
    We can also investigate the role of each country on the trade of soybeans by looking on the figures of how much each of them negotiated over the years (1996-2022).
    """
    trade = pd.DataFrame()
    trade["exports"] = data.groupby(["Source"]).sum().sort_values(by=['trade_value'], ascending=False)["trade_value"]
    trade["imports"] = data.groupby(["Target"]).sum().sort_values(by=['trade_value'], ascending=False)["trade_value"]

    """
    The summary of the main exporters and importers is presented below.
    """
    st.subheader("Main exporters")
    st.write(trade.sort_values(by=['exports'], ascending=False).head()["exports"])

    st.subheader("Main importers")
    st.write(trade.sort_values(by=['imports'], ascending=False).head()["imports"])

    trade["ln_exports"] = np.log(trade["exports"])
    trade["ln_imports"] = np.log(trade["imports"])


    fig = px.scatter(trade, x='ln_exports', y='ln_imports', hover_name=trade.index)

    # calculate averages
    x_avg = trade['ln_exports'].mean()
    y_avg = trade['ln_imports'].mean()

    # add horizontal and vertical lines
    fig.add_vline(x=x_avg, line_width=1, opacity=0.5)
    fig.add_hline(y=y_avg, line_width=1, opacity=0.5)

    # set x limits
    adj_x = max((trade['ln_exports'].max() - x_avg), (x_avg - trade['ln_exports'].min())) * 1.1
    lb_x, ub_x = (x_avg - adj_x, x_avg + adj_x)
    fig.update_xaxes(range = [lb_x, ub_x])

    # set y limits
    adj_y = max((trade['ln_imports'].max() - y_avg), (y_avg - trade['ln_imports'].min())) * 1.1
    lb_y, ub_y = (y_avg - adj_y, y_avg + adj_y)
    fig.update_yaxes(range = [lb_y, ub_y])

    # update x tick labels
    axis = ['Low', 'High']     
    fig.update_layout(
        xaxis_title='Ln Vol. Imports',
        xaxis = dict(
            tickmode = 'array',
            tickvals = ([(x_avg - adj_x / 2), (x_avg + adj_x / 2)]),
            ticktext = axis
        )
        )

    # update y tick labels
    fig.update_layout(
        yaxis_title='Ln Vol. Exports',
        yaxis = dict(
            tickmode = 'array',
            tickvals = ([(y_avg - adj_y / 2), (y_avg + adj_y / 2)]),
            ticktext = axis,
            tickangle=270
            )
        ) 

    fig.update_layout(margin=dict(t=50, l=5, r=5, b=50),
    title={'text': 'Soybeans trade: Sellers x Buyers',
            'font_size': 20,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig.add_annotation(x=trade.ln_exports.min()+1, y=trade.ln_imports.min(),
                    text="Low imports, low exports",
                    showarrow=False)
    fig.add_annotation(x=trade.ln_exports.min()+1, y=trade.ln_imports.max()+2,
                    text="High imports, low exports",
                    showarrow=False)
    fig.add_annotation(x=trade.ln_imports.mean(), y=trade.ln_imports.min(),
                    text="Low imports, high exports",
                    showarrow=False)
    fig.add_annotation(x=trade.ln_imports.mean(), y=trade.ln_imports.max()+2,
                    text="High imports, high exports",
                    showarrow=False)

    # Plot!
    st.plotly_chart(fig, use_container_width=True)

display_analysis(option)
