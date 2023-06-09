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

st.title(f'Balassa Index on Soybeans Trade')
"""
The current analysis seeks to identify the most important products in the world trade network by considering the Balassa Index.

This index was firstly introduced by Bela Balassa in 1965 and it is used to measure the comparative advantage of a country in a certain product. The index is calculated as follows:

Balassa Index = (Xij / Xi) / (Xwj / Xw)

Where Xij is the exports of product j from country i, Xi is the total exports of country i, Xwj is the exports of product j from the world, and Xw is the total exports of the world.

It is ussualy said that a value greater than 1 indicates that the country has a comparative advantage in that product, while a value less than 1 indicates that the country has a comparative disadvantage in that product.

The threshold might varies, however it is important to understand that the higher the value, the higher the comparative advantage of a country in a certain product.

"""


""""
To start with, we will import the data from the United Nations Commodity Trade Statistics Database (UN Comtrade). To facilitate the proccess I have already downloaded the data and uploaded to my GitHub repository.

The data compromises four datasets covering the period from 1996 to 2022. 
The first dataset contains the soybeans trade data, the second one contains the country exports, the third one contains the world exports, and the last one contains the soybeans world trade.
The first two datasets will be used to calculate the numerator of the Balassa Index, while the last two be part of the denominator. 
"""
soy_data = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soytrade_dataset.csv')
country_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/country_exports.csv')
world_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/world_exports.csv')
soy_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soy_world_trade.csv')


code = ''' 
soy_data = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soytrade_dataset.csv')
country_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/country_exports.csv')
world_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/world_exports.csv')
soy_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soy_world_trade.csv')
'''

st.code(code, language='python')

"""
Bellow we can examine the content of these datasets, I print the first 10 lines of each.
"""
st.subheader("Soy exporters data")
st.write(soy_data.head())
st.subheader("Country exports data")
st.write(country_exports.head())
st.subheader("World exports data")
st.write(world_exports.head())
st.subheader("Soy world trade data")
st.write(soy_exports.head())

"""
As we can see the soy data exports data is stratified by each partner, however we need to aggregate this data by country. A simple way to do this is by using the groupby function.
"""

code = '''
soy_country_exports = soy_data.groupby(['Source', 'Year'], as_index=False).sum().loc[:, ['Source', 'Year', 'trade_value']]
'''
st.code(code, language='python')
soy_country_exports = soy_data.groupby(['Source', 'Year'], as_index=False).sum().loc[:, ['Source', 'Year', 'trade_value']]

"""
A quick visualization on the new dataset shows that we have the total exports of soybeans by country and year. 
"""
st.write(soy_country_exports.head())

"""
Usually I start calculating the easiest part of the problems I have, which in this case I think is the denominator, since we just have to calculate the total exports of soybeans relative to total export by year.
The code bellow merges the two datasets into a new one called denominator, and calculates the total exports of soybeans relative to total export by year. 
The collumm 'Share' is the one we are interested in, since it represents the percentage of soybeans in the total exports and will later on be used to calculate the Balassa Index.
"""

denominator = world_exports.merge(soy_exports, on='Year').loc[:, ['Year', 'TradeValue in 1000 USD_x', 'TradeValue in 1000 USD_y']]
denominator.columns = ['Year', 'World Exports', 'Soy Exports']
denominator['Share'] = denominator['Soy Exports'] / denominator['World Exports']

code = ''' 
# merge the two datasets and select only the columns we need
denominator = world_exports.merge(soy_exports, on='Year').loc[:, ['Year', 'TradeValue in 1000 USD_x', 'TradeValue in 1000 USD_y']]
# rename the columns
denominator.columns = ['Year', 'World Exports', 'Soy Exports']
# calculate the percentage
denominator['Share'] = denominator['Soy Exports'] / denominator['World Exports']
'''
st.code(code, language='python')

"""
Half the work is done, now it is time to get the numerator of the Balassa Index. So we have to get the total exports of soybeans by country and year, and then divide it by the total exports of the country by year.

As we could see with a quick observation in the soy exports by country, there are years with no data, and also not all countries have soybeans exports. 
Hence one simple solution is to add the country exports data into the soy country exports dataframe, since we do not need to keep track of what data is missing.
A simple for loop will do the job, however admittedly it is not the most efficient way to do it.
Additionally, to make the analysis easier, I will rename the columns of the soy country exports dataframe.
With the new column 'Country Export' we can calculate the numerator of the Balassa Index, by dividing the 'Trade Value' by the 'Country Export'.
"""

for row in range(len(soy_country_exports)):
    soy_country_exports.loc[row, 'Country Export'] = country_exports.loc[(country_exports['ReporterISO3'] == soy_country_exports.loc[row]['Source']) & (country_exports['Year'] == soy_country_exports.loc[row]['Year']), 'TradeValue in 1000 USD'].values[0]

code = ''' 
for row in range(len(soy_country_exports)):
    soy_country_exports.loc[row, 'Country Export'] = country_exports.loc[(country_exports['ReporterISO3'] == soy_country_exports.loc[row]['Source']) & (country_exports['Year'] == soy_country_exports.loc[row]['Year']), 'TradeValue in 1000 USD'].values[0]

soy_country_exports.columns = ['Country', 'Year', 'Trade Value', 'Country Export']

soy_country_exports['Share'] = soy_country_exports['Trade Value'] / soy_country_exports['Country Export']
'''
st.code(code, language='python')

soy_country_exports.columns = ['Country', 'Year', 'Trade Value', 'Country Export']

soy_country_exports['Share'] = soy_country_exports['Trade Value'] / soy_country_exports['Country Export']

"""
We can visualize the new dataframe bellow.
"""

st.write(soy_country_exports.head())

"""
Now that we have the numerator and the denominator, we can calculate the index. 
I merge the two dataframes based on the 'Year' column, and rename the columns to make it easier to understand. 
Consequently we will have the Ballassa index, since it is just the quotient of these two values.
"""

code = '''
# Merge the two DataFrames based on the "Year" column
soy_country_exports = soy_country_exports.merge(denominator[['Year', 'Share']], on='Year', how='left')

# Rename the "Share" column from the denominator DataFrame
soy_country_exports.rename(columns={'Share_y': 'Denominator', 'Share_x': 'Numerator'}, inplace=True)
soy_country_exports['Balassa Index'] = soy_country_exports['Numerator'] / soy_country_exports['Denominator']
'''

soy_country_exports = soy_country_exports.merge(denominator[['Year', 'Share']], on='Year', how='left')
soy_country_exports.rename(columns={'Share_y': 'Denominator', 'Share_x': 'Numerator'}, inplace=True)
soy_country_exports['Balassa Index'] = soy_country_exports['Numerator'] / soy_country_exports['Denominator']

"""
We can the final dataframe bellow.
"""
st.write(soy_country_exports)


"""
A last step is to visualize the top 5 countries with the highest Balassa Index. And observe the trend they have over the years.
"""
def get_top_balassa_index(df, top=10):
    return df.groupby('Country').mean().sort_values(by='Balassa Index', ascending=False).head(top).index.values


number = 5
top = get_top_balassa_index(soy_country_exports, top=number)

fig = px.line(soy_country_exports.loc[soy_country_exports['Country'].isin(top), :], x='Year', y='Balassa Index', color='Country', symbol="Country")

# Adjust the figure layout
fig.update_layout(
    width=800,  # Adjust the width as desired
    height=600,  # Adjust the height as desired
)

fig.update_xaxes(range=[soy_country_exports['Year'].min(), soy_country_exports['Year'].max() + 2])  # Assumes the data is sorted by year
st.plotly_chart(fig)

plt.figure(figsize=(10, 6))

grouped_df = soy_country_exports.loc[soy_country_exports['Country'].isin(top), :].groupby('Country')

for country, data in grouped_df:
    plt.plot(data['Year'], data['Balassa Index'], label=country)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Balassa Index')
plt.title(f'Top {number} Balassa Index for Soybeans')

plt.xlim(soy_country_exports['Year'].min() + 1, soy_country_exports['Year'].max() + 1)

# Add legend
plt.legend()

# Show the chart
plt.show()