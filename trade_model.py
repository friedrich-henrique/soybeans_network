# %%
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib import colors
from os import listdir
from os.path import isfile, join
import numpy as np

# %%
#soy_trade = pd.read_excel('data/soytrade.xlsx')

soy_trade = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soytrade.csv?token=GHSAT0AAAAAACAIZJ3JI6FZBAGVWOJC63GWZCWBUCA')
country_size = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/countries_size.csv?token=GHSAT0AAAAAACAIZJ3J3FI7LT44R7AGRWGAZCWBV7Q')

# %%
soy_trade.head()

# %%
country_size.head()

# %%
country_size.drop([0,1,2], inplace=True)
country_size.columns = country_size.iloc[0]
country_size.drop([3], inplace=True)
country_size.drop(columns=['Indicator Name','Indicator Code', 'Country Name', 1960.0], inplace=True)

# %%
sizes = country_size.melt(id_vars=["Country Code"],
                  var_name="Year",
                  value_name="Area")

# %%
sizes["Year"] = sizes["Year"].apply(lambda x: int(x))

# %%
sizes.head()

# %%
soy_trade.rename(columns={"1201 in 1000 USD ": "trade_value"}, inplace=True)
soy_trade['ln_trade_value'] = soy_trade['trade_value'].apply(lambda x: np.log(x))

# %%
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


# %%
def append_sizes_to_trade_table(sizes_table, trade_table, attribute_name):
    count = 0
    progress = 0
   
    for row in range(len(soy_trade)):
        source_area_slice = sizes_table[(sizes["Country Code"] == trade_table.iloc[row, 0]) & (sizes["Year"] == trade_table.loc[row]["Year"])]
        target_area_slice = sizes_table[(sizes["Country Code"] == trade_table.iloc[row, 1]) & (sizes["Year"] == trade_table.loc[row]["Year"])]
        if len(source_area_slice) > 0:
            trade_table.at[row, f"Source {attribute_name}"] = source_area_slice.iloc[0][attribute_name]
        else: 
            trade_table.at[row, f"Source {attribute_name}"] = None
        if len(target_area_slice) > 0:
            trade_table.at[row, f"Target {attribute_name}"] = target_area_slice.iloc[0][attribute_name]
        else:
            trade_table.at[row, f"Target {attribute_name}"] = None
        count += 1
        current_progress = count / len(soy_trade) * 100
        if current_progress >= 1:
            if current_progress >= progress + 10:
                progress = int(current_progress // 10) * 10
                print(f"{progress}% done")

# %%
data = add_attributes(soy_trade, sizes, "Area")

# %%
data.head()

# %%
# relationship between trade value and area of brazilian partners   
g = sns.jointplot(x="ln_trade_value", y="Target Area", data=data[ (data['Source'] == 'BRA') & (data['Year'] == 2017)],
                  kind="reg", truncate=False,
                  color="m")

# %%
trade_brazil_2017 = data[ (data['Source'] == 'BRA') & (data['Year'] == 2017)]

# %%
import plotly.express as px

fig = px.scatter(trade_brazil_2017, y="Target Area", x="trade_value", text="Target", log_x=True, size_max=100, color="trade_value", trendline="ols")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Country Partner Area', title_x=0.5)
fig.show()

# %%
sum_trade = data[soy_trade['Year'] == 2020].groupby(['Source', 'Source Area'], as_index=False)['trade_value'].sum().sort_values(by=['trade_value'], ascending=False)

# %%
sum_trade.head()

# %%
fig = px.scatter(sum_trade, y="Source Area", x="trade_value", text="Source", log_x=True, size_max=100, color="trade_value", trendline="ols")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Exports x Source Area', title_x=0.5)
fig.show()

# %%
ex_rate = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/Exchange%20Rates.csv?token=GHSAT0AAAAAACAIZJ3J2IXNVLAXPYSOMDTOZCWB2RQ', skiprows=2)
ex_rate.rename({'Unnamed: 0': 'Country Code'}, inplace=True, axis=1)
ex_rate.head()

# %%
ex_rate = ex_rate.melt(id_vars=["Country Code"],
                  var_name="Year",
                  value_name="Exchange Rate")

# %%
ex_rate.head()

# %%
ex_rate["Year"] = ex_rate["Year"].apply(lambda x: int(x))

# %%
data = add_attributes(soy_trade, ex_rate, "Exchange Rate")

# %%
data.head()

# %%
print (data['Source Exchange Rate'].isnull().sum(), data['Target Exchange Rate'].isnull().sum())

# %%
# Subsitute nan of eu countries with EMU exchange rate
## EMU = European Monetary Union (Euro)
eu_countries = ['AUT', 'BEL', 'CYP', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN', 'ESP']

for year in range(1999, data['Year'].max()+1):
    for country in eu_countries:
        ex_rate_value = ex_rate[(ex_rate["Year"] == year) & (ex_rate["Country Code"] == 'EMU')]['Exchange Rate'].values[0]
        data.loc[(data["Year"] == year) & (data["Source"].isin(eu_countries)), 'Source Exchange Rate'] = ex_rate_value
        data.loc[(data["Year"] == year) & (data["Target"].isin(eu_countries)), 'Target Exchange Rate'] = ex_rate_value

# %%
print (data['Source Exchange Rate'].isnull().sum(), data['Target Exchange Rate'].isnull().sum())

# %%
data['Source Exchange Rate'] = [0 if i is None else float(str(i).replace(",", "")) for i in data["Source Exchange Rate"]]
data['Target Exchange Rate'] = [0 if i is None else float(str(i).replace(",", "")) for i in data["Target Exchange Rate"]]

# %%
data["Ex Rate Source / Target"] = data["Source Exchange Rate"] / data["Target Exchange Rate"]
data["Ex Rate Target / Source"] = data["Target Exchange Rate"] / data["Source Exchange Rate"]

# %%
consumer_price = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/Consumer%20Price%20Index.csv?token=GHSAT0AAAAAACAIZJ3ILPT5RHHYRPV7PCQCZCWB3UA', skiprows=2)

consumer_price.rename({'Unnamed: 0': 'Country Code'}, inplace=True, axis=1)

# %%
consumer_price.head()

# %%
consumer_price = consumer_price.melt(id_vars=["Country Code"],
    var_name="Year",
    value_name="Consumer Price Index")

consumer_price.head()

# %%
consumer_price["Year"] = consumer_price["Year"].apply(lambda x: int(x))

# %%
data = add_attributes(data, consumer_price, "Consumer Price Index")

# %%
data.head()

# %%
data['Source Consumer Price Index'] = [0 if i is None else float(str(i).replace(",", "")) for i in data["Source Consumer Price Index"]]
data['Target Consumer Price Index'] = [0 if i is None else float(str(i).replace(",", "")) for i in data["Target Consumer Price Index"]]

# %%
data["exchange_rate ijt"] = data["Ex Rate Target / Source"] * (data["Source Consumer Price Index"] / data["Target Consumer Price Index"])

# %%
gdp_data = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/Gross%20Domestic%20Product.csv?token=GHSAT0AAAAAACAIZJ3J7OIU3VFDAPUG7342ZCWB4BQ',skiprows=2)

gdp_data.rename({'Unnamed: 0': 'Country Code'}, inplace=True, axis=1)

gdp_data = gdp_data.melt(id_vars=["Country Code"],
    var_name="Year",
    value_name="GDP")
gdp_data["Year"] = gdp_data["Year"].apply(lambda x: int(x))
gdp_data.head()

# %%
data = add_attributes(data, gdp_data, "GDP")

# %%
data['Source GDP'] = [0 if i is None else float(str(i).replace(",", "")) for i in data["Source GDP"]]
data['Target GDP'] = [0 if i is None else float(str(i).replace(",", "")) for i in data["Target GDP"]]

# %%
data.replace(0, np.nan, inplace=True)

# %%
data["level of output ijt"] = (np.log(data["Source GDP"]) + np.log(data["Target GDP"])) / 2

# %%
data.dropna().head()

# %%
fig = px.scatter(data.dropna(), y="trade_value", x="level of output ijt", color="trade_value", trendline="ols")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Trade x Output', title_x=0.5)
fig.show()

# %%
fig = px.scatter(data.dropna(), y="trade_value", x="level of output ijt", log_y=True, size_max=100, color="trade_value", trendline="ols")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Trade x GDP', title_x=0.5)
fig.show()

# %%
trade = pd.DataFrame()
trade["exports"] = data.groupby(["Source"]).sum().sort_values(by=['trade_value'], ascending=False)["trade_value"]
trade["imports"] = data.groupby(["Target"]).sum().sort_values(by=['trade_value'], ascending=False)["trade_value"]

# %%
trade.sort_values(by=['exports'], ascending=False).head()

# %%
trade["ln_exports"] = np.log(trade["exports"])
trade["ln_imports"] = np.log(trade["imports"])

# %%
trade.ln_exports.min(), trade.ln_exports.max()

trade.ln_imports.min(), trade.ln_imports.max()

# %%
plt.figure(figsize=(12,8))
sns.scatterplot(data=trade, x='ln_exports', y='ln_imports')
plt.title(f"Soybeans trade: Sellers x Buyers")
plt.xlabel("Ln Vol. Exports")
plt.ylabel("Ln Vol. Imports")

#Country names
for i in range(trade.shape[0]):
          plt.text(trade.ln_exports[i], y=trade.ln_imports[i], s=trade.iloc[i].name, mouseover=True, fontsize=10)

# Benchmark Mean values          
plt.axhline(y=trade.ln_exports.mean(), color='k', linestyle='--', linewidth=1)           
plt.axvline(x=trade.ln_imports.mean(), color='k',linestyle='--', linewidth=1) 

#Quadrant Marker          
plt.text(x=trade.ln_exports.min()-1, y=trade.ln_imports.min(), s="Low imports, low exports",alpha=0.5,fontsize=10, color='black')
plt.text(x=trade.ln_exports.min()-1, y=trade.ln_imports.max()+0.4, s="High imports, low exports",alpha=0.5,fontsize=10, color='black')
plt.text(x=trade.ln_imports.mean()+0.1, y=trade.ln_imports.min(), s="Low imports, high exports", alpha=0.5,fontsize=10, color='black')
plt.text(x=trade.ln_imports.mean()+0.1, y=trade.ln_imports.max()+0.4, s="High imports, high exports", alpha=0.5,fontsize=10, color='black')

plt.show()