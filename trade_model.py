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

# %%
soy_trade = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soytrade.csv')
country_size = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/countries_size.csv')
ex_rate = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/Exchange%20Rates.csv', skiprows=2)
gdp_data = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/Gross%20Domestic%20Product.csv',skiprows=2)
consumer_price = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/Consumer%20Price%20Index.csv', skiprows=2)
st.title('Soybeans Trade Analysis')

soy_trade.rename(columns={"1201 in 1000 USD ": "trade_value"}, inplace=True)
soy_trade['ln_trade_value'] = soy_trade['trade_value'].apply(lambda x: np.log(x))


country_size.drop([0,1,2], inplace=True)
country_size.columns = country_size.iloc[0]
country_size.drop([3], inplace=True)
country_size.drop(columns=['Indicator Name','Indicator Code', 'Country Name', 1960.0], inplace=True)


gdp_data.rename({'Unnamed: 0': 'Country Code'}, inplace=True, axis=1)

ex_rate.rename({'Unnamed: 0': 'Country Code'}, inplace=True, axis=1)

consumer_price.rename({'Unnamed: 0': 'Country Code'}, inplace=True, axis=1)
consumer_price = consumer_price.melt(id_vars=["Country Code"],
    var_name="Year",
    value_name="Consumer Price Index")
consumer_price["Year"] = consumer_price["Year"].apply(lambda x: int(x))
# %%
sizes = country_size.melt(id_vars=["Country Code"],
                  var_name="Year",
                  value_name="Area")

# %%
sizes["Year"] = sizes["Year"].apply(lambda x: int(x))

def add_attributes(trade_table, sizes_table, attribute_name):
    trade_table = trade_table.copy()
    sizes_table = sizes_table.copy()
    sizes_table = sizes_table[sizes_table["Year"] >= trade_table["Year"].min()] 
    trade_table.reset_index(drop=True, inplace=True)
    sizes_table.reset_index(drop=True, inplace=True)
    count = 0
    progress = 0
   
    for row in range(len(trade_table)):
        source_area_slice = sizes_table[(sizes_table["Country Code"] == trade_table.iloc[row, 0]) & (sizes_table["Year"] == trade_table.loc[row]["Year"])]
        target_area_slice = sizes_table[(sizes_table["Country Code"] == trade_table.iloc[row, 1]) & (sizes_table["Year"] == trade_table.loc[row]["Year"])]
        if len(source_area_slice) > 0:
            trade_table.at[row, f"Source {attribute_name}"] = source_area_slice.iloc[0][attribute_name]
        else: 
            trade_table.at[row, f"Source {attribute_name}"] = None

        if len(target_area_slice) > 0:
            trade_table.at[row, f"Target {attribute_name}"] = target_area_slice.iloc[0][attribute_name]
        else:
            trade_table.at[row, f"Target {attribute_name}"] = None
        count += 1
        current_progress = count / len(trade_table) * 100
        if current_progress >= 1:
            if current_progress >= progress + 10:
                progress = int(current_progress // 10) * 10
                print(f"{progress}% done")
    return trade_table

data = add_attributes(soy_trade, sizes, "Area")

"""
To start with the analysis, we can visualize the soybeans trade data the respective area of each country.
"""
st.write(data)

# relationship between trade value and area of brazilian partners   
st.header("Relationship between 2017 trade value and area of brazilian partners")

"""
Here the trade data is presented by the log of the trade value and the area of the partner countries.
"""
#### create space for the replacement of the chart
chart_placeholder = st.empty()
chart_placeholder.pyplot()
chart = sns.jointplot(x="ln_trade_value", y="Target Area", data=data[ (data['Source'] == 'BRA') & (data['Year'] == 2017)],
                  kind="reg", truncate=False,
                  color="m")
chart_placeholder.pyplot(chart, use_container_width=True)

"""
In order to inspect the 'outliers' of the previous chart, one can visualize the data in an interactive way.
"""
trade_brazil_2017 = data[ (data['Source'] == 'BRA') & (data['Year'] == 2017)]
fig = px.scatter(trade_brazil_2017, y="Target Area", x="trade_value", text="Target", log_x=True, size_max=100, color="trade_value", trendline="ols")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Country Partner Area', title_x=0.5)
# Plot!
st.plotly_chart(fig, use_container_width=True)


"""
Additionally we can investigate the relationship of the country's area and the trade volume of soybeans in 2020
"""

sum_trade = data[soy_trade['Year'] == 2020].groupby(['Source', 'Source Area'], as_index=False)['trade_value'].sum().sort_values(by=['trade_value'], ascending=False)

"""
Here are the main exporters in 2020.
"""
st.table(sum_trade.head())

"""
We can see the correlation between the trade value and the area of the country in the following chart.
"""
fig = px.scatter(sum_trade, y="Source Area", x="trade_value", text="Source", log_x=True, size_max=100, color="trade_value", trendline="ols")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Exports x Source Area', title_x=0.5)
# Plot!
st.plotly_chart(fig, use_container_width=True)

st.header("Countrys participation on trade")
"""
we can also investigate the role of each country on the trade of soybeans by looking on the figures of how much each of them negotiated over the years.
"""
trade = pd.DataFrame()
trade["exports"] = data.groupby(["Source"]).sum().sort_values(by=['trade_value'], ascending=False)["trade_value"]
trade["imports"] = data.groupby(["Target"]).sum().sort_values(by=['trade_value'], ascending=False)["trade_value"]

"""
The summary of the main exporters and importers is presented below.
"""
st.subheader("Main exporters")
st.write(trade.sort_values(by=['exports'], ascending=False).head())

st.subheader("Main importers")
st.write(trade.sort_values(by=['imports'], ascending=False).head())

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