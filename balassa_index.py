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

"""
The current analysis seeks to identify the most important products in the world trade network by considering the Balassa Index.

This index was firstly introduced by Bela Balassa in 1965 and it is used to measure the comparative advantage of a country in a certain product. The index is calculated as follows:

Balassa Index = (Xij / Xi) / (Xwj / Xw)

    where Xij is the exports of product j from country i, Xi is the total exports of country i, Xwj is the exports of product j from the world, and Xw is the total exports of the world.

It is ussualy said that a value greater than 1 indicates that the country has a comparative advantage in that product, while a value less than 1 indicates that the country has a comparative disadvantage in that product.

The threshold might varies, however it is important to understand that the higher the value, the higher the comparative advantage of a country in a certain product.

"""

soy_data = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soytrade_dataset.csv')
country_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/country_exports.csv')
world_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/world_exports.csv')
soy_exports = pd.read_csv('https://raw.githubusercontent.com/friedrich-henrique/datasets_soybeans_research/main/soy_world_trade.csv')