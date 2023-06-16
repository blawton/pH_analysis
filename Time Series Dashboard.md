```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.fft as fft
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.ndimage import median_filter
from statsmodels.tsa.seasonal import STL
from datetime import datetime
```

```python
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
```

# Loading the Data


## pH Data Loaded by Default

```python
ph=pd.read_csv("../Data Visualization and Analytics Challenge/Data/processed_ph_1.24.csv", index_col=0)
ph.head()
```

```python
ph.columns
```

```python
#Renaming pH columns or uniform lineup
ph.rename(columns={"ActivityStartTime/Time":"time", "ActivityDepthHeightMeasure/MeasureValue":"depth", "ResultMeasureValue":"pH"}, inplace=True)
ph.drop("year", axis=1, inplace=True)
ph.head()
```

## Choosing Other Data Sources

```python
main=pd.read_csv("../Data Visualization and Analytics Challenge/Data/CT_DEEP_Post_2010_1.12.2023.csv", index_col=0)
main.head()
```

```python
main.rename(columns={"Time ON Station":"Time"}, inplace=True)
main.head()
```

```python
#Sorting values
main.sort_values(["station name", "Date", "depth"], inplace=True)
main.head()
```

```python
stations=pd.read_csv("../OneDrive - Environmental Protection Agency (EPA)/Downloads/Basin_Labelled_Stations.csv", index_col=[0])
stations
```

```python
#CTDEEP year-round stations
year_round = ["A4", "B3", "C1", "C2", "D3", "E1", "09", "15", "F2", "F3", "H2", "H4", "H6", "I2", "J2", "K2", "M3"]
year_round = ["CT_DEP01_WQX-" + x for x in year_round]
print(len(year_round))
year_round
```

```python
#Making format uniform between WQX ph and do
main["MonitoringLocationIdentifier"]="CT_DEP01_WQX-" + main["station name"].astype(str)
```

```python
#Dates
main["Date"]=pd.to_datetime(main["Date"])
main["Year"]=main["Date"].dt.year
main["day_of_year"]=main["Date"].dt.dayofyear
```

```python
#Getting basins
main=main.merge(stations[["MonitoringLocationIdentifier", "loc", "area"]], how="left", on="MonitoringLocationIdentifier")
```

```python
#Investigating missing stations
pd.unique(main.loc[main["area"].isna(), "MonitoringLocationIdentifier"])
```

```python
#Station D3
main=main.loc[~((main["cruise name"]=="WQMAR20")& (main["station name"]=="D3"))]
```

## Merging ph with covariates

```python
covariates=["Dissolved Oxygen", "salinity", "temperature"]
```

```python
#Making list of columns to be merged
column_list=[column for column in main.columns if column in ph.columns]
column_list += covariates

#Avoiding duplication of pH
column_list.remove("pH")

print(column_list)
agg=pd.concat([ph, main[column_list]])
agg.head()
```

# Time Series Analysis

```python
#Graphing param
plt.rcParams["figure.figsize"]=(21, 10)
```

```python
#Ensuring Numeric values and Datetime format
for cov in covariates:
    agg[cov]=pd.to_numeric(agg[cov])
    
agg["Date"]=pd.to_datetime(agg["Date"])
```

```python
def tsa(data, cov):
    for basin in np.arange(1, 6):
        for st in year_round:

            working=data.loc[(data["area"]==basin) & (data["MonitoringLocationIdentifier"]==st)]

            if (len(working)>0):
                
                working_cov=working.dropna(subset=cov, axis=0)
                working_ph=working.dropna(subset="pH", axis=0)
                
                #Monthly sampling
                monthly_cov=working_cov.set_index("Date")[cov].resample('M').mean().ffill()
                monthly_ph=working_ph.set_index("Date")["pH"].resample('M').mean().ffill()

                #Models
                res_cov = STL(monthly_cov).fit()
                res_ph = STL(monthly_ph).fit()

                #graphing
                seasonal_cov=res_cov.seasonal/abs(min(res_cov.seasonal))
                seasonal_ph=res_ph.seasonal/abs(min(res_ph.seasonal))
                seasonal_cov.plot(label=cov)
                seasonal_ph.plot(label="pH")
                plt.legend(loc="upper right", fontsize=20)
                ax=plt.gca()
                late_summer=[datetime.strptime('Aug 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(data["Year"]))]
                new_year=[datetime.strptime('Jan 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(data["Year"]))]
                ax.set_xticks(late_summer, minor=True)
                ax.set_xticks(new_year, major=True)
                ax.xaxis.grid(True, linewidth=2, which='minor', color="r")
                ax.xaxis.grid(True, linewidth=2, which='major')
                ax.set_title("Basin " + str(basin) + ": " + st[-2:], fontsize="xx-large")
                plt.show()
```

```python
tsa(agg, "Dissolved Oxygen")
```
![download](https://github.com/blawton/pH_analysis/assets/46683509/9e6ac80f-2fe1-4b60-a8a3-bad519e7efbc)

![download](https://github.com/blawton/pH_analysis/assets/46683509/be3508d9-ec21-4150-9464-b44faceb3db2)

![download](https://github.com/blawton/pH_analysis/assets/46683509/c560c4af-8f96-465b-a953-1c2de2db03e2)

![download](https://github.com/blawton/pH_analysis/assets/46683509/42c21631-109f-4582-b578-d0c49295abcd)

![download](https://github.com/blawton/pH_analysis/assets/46683509/3e1df251-91ea-4155-b21a-61c37cb675e2)

![download](https://github.com/blawton/pH_analysis/assets/46683509/d5ee9196-7c93-4560-b6c5-f9b468bd7fb4)

![download](https://github.com/blawton/pH_analysis/assets/46683509/6deda64a-eecb-4dea-9de6-af5b01be44f2)

![download](https://github.com/blawton/pH_analysis/assets/46683509/679b0e65-cf98-4eb4-9231-14bb731e1213)

![download](https://github.com/blawton/pH_analysis/assets/46683509/8d46bcfe-38e9-43e4-a9dd-061bc171469f)

![download](https://github.com/blawton/pH_analysis/assets/46683509/6f676bae-8497-4519-aa88-35ae5079866c)

![download](https://github.com/blawton/pH_analysis/assets/46683509/8c4a38b1-db4d-4398-b31e-8579e0058bce)

![download](https://github.com/blawton/pH_analysis/assets/46683509/dfc36abf-c05b-49cf-9895-cd7ea6cf4fff)

![download](https://github.com/blawton/pH_analysis/assets/46683509/b7438cd4-0bef-4e38-9173-4db7f0a22449)

![download](https://github.com/blawton/pH_analysis/assets/46683509/5532603d-8771-4846-b483-5a2270a1d99a)

![download](https://github.com/blawton/pH_analysis/assets/46683509/89fea085-56ad-4912-bbf1-6689f6bd5b60)

![download](https://github.com/blawton/pH_analysis/assets/46683509/cf699dc3-12f9-41a8-8ae5-2e5ba49fed75)


```python
#Seasonal param
param="salinity"
seasonal_focus=7
```

```python
seasons = dict(zip(np.arange(1,13), ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]))

```

```python
#Yearly Approach
seasonal=agg.copy(deep=True)
seasonal["Month"]=seasonal["Date"].dt.month
seasonal=seasonal.loc[seasonal["Month"]==7]
grouped=seasonal.groupby(["area", "Year"])[param].mean().unstack(level=[1])
grouped.index.name="Basin"
grouped.index=grouped.index.astype(int)
grouped
```

```python
#Simple Seasonal Approach
seasonal=agg.copy(deep=True)
seasonal["Month"]=seasonal["Date"].dt.month
grouped=seasonal.groupby(["area", "Month"])[param].mean().unstack(level=[1])
grouped.index.name="Basin"
grouped.index=grouped.index.astype(int)
grouped.columns=seasons.values()
grouped
```

Next step is to look for a correlation with water inflows as a way of seeing if it's driven by aragonite saturation or salinity
