import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


indicators = pd.read_csv('../dataset/Indicators.csv')
countries = pd.read_csv('../dataset/Country.csv')
series = pd.read_csv('../dataset/Series.csv')


countries['Region'] = countries['Region'].fillna('Unknown')
countries['IncomeGroup'] = countries['IncomeGroup'].fillna('Unknown')

indicators.head()

#SI.POV.DDAY - Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)
#SL.EMP.TOTL.SP.ZS - The Employment to Population Ratio is the percentage of a country’s population aged 15 years and older that is employed.
#SE.ADT.LITR.ZS - Literacy rate (%)
#NY.GDP.PCAP.CD - GDP per capita (current US$)
relevant_indicators = ['SI.POV.DDAY', 'NY.GDP.PCAP.CD', 'SL.EMP.TOTL.SP.ZS', 'SE.ADT.LITR.ZS']
filtered_data = indicators[indicators['IndicatorCode'].isin(relevant_indicators)]
print(filtered_data['IndicatorCode'].unique())
data_grouped = filtered_data.groupby(['Year', 'IndicatorCode'])['Value'].mean().reset_index()

pivot_data = data_grouped.pivot(index='Year', columns='IndicatorCode', values='Value')
correlations = pivot_data.corr()
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title("Correlation Between Indicators")
plt.show()


pivot_data.head()
print(pivot_data.columns)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pivot_data, x='NY.GDP.PCAP.CD', y='SI.POV.DDAY')
plt.title("GDP Per Capita vs Poverty Headcount Ratio")
plt.xlabel("GDP Per Capita (US$)")
plt.ylabel("Poverty Headcount Ratio (%)")
plt.show()


sns.regplot(data=data_grouped[data_grouped['IndicatorCode'] == 'SI.POV.DDAY'], 
            x='Year', y='Value', scatter_kws={'s':10})
plt.title("Trendline for Poverty Headcount Ratio Over Time")
plt.show()

