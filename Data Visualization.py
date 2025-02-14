import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


indicators = pd.read_csv('dataset/Indicators.csv')


indicators.head()

#SI.POV.DDAY - Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)
#SL.EMP.TOTL.SP.ZS - The Employment to Population Ratio is the percentage of a country’s population aged 15 years and older that is employed.
#SE.ADT.LITR.ZS - Literacy rate (%)
#relevant_indicators = ['SI.POV.DDAY', 'NY.GDP.PCAP.CD', 'SL.EMP.TOTL.SP.ZS', 'SE.ADT.LITR.ZS']
relevant_indicators = ['SI.POV.DDAY', 'SL.EMP.TOTL.SP.ZS', 'SE.ADT.LITR.ZS']
filtered_data = indicators[indicators['IndicatorCode'].isin(relevant_indicators)]
filtered_data.head()

#VISUALIZATION
data_grouped = filtered_data.groupby(['Year', 'IndicatorCode' , 'IndicatorName'])['Value'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=data_grouped, x='Year', y='Value', hue='IndicatorName')
plt.title("Trends of Poverty and Related Indicators Over Time")
plt.ylabel("Value")
plt.xlabel("Year")
plt.legend(title="Indicator Code")
plt.show()
#VISUALIZATION


#NY.GDP.PCAP.CD - GDP per capita (current US$)
relevant_indicators = ['NY.GDP.PCAP.CD']
filtered_data = indicators[indicators['IndicatorCode'].isin(relevant_indicators)]
filtered_data.head()

#VISUALIZATION
data_grouped = filtered_data.groupby(['Year', 'IndicatorCode' , 'IndicatorName'])['Value'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=data_grouped, x='Year', y='Value', hue='IndicatorName')
plt.title("Trends of Poverty with GDP Rate Over Time")
plt.ylabel("Value")
plt.xlabel("Year")
plt.legend(title="Indicator Code")
plt.show()
#VISUALIZATION


