```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```python
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
```

# Loading the Data

```python
agg_data=pd.read_csv("cleaned_aggregated_pH_data.csv")
agg_data.head()
```

```python
counted=agg_data.groupby(["year", "OrganizationIdentifier"])["ResultMeasureValue"].count()
counted.unstack(0).head(15)
```

```python
#Selecting organizations with LIS presence
orgs=["31ISC2RS_WQX", "CT_DEP01_WQX", "USGS-CT", "USGS-NY"]
trend_data=agg_data.loc[agg_data["OrganizationIdentifier"].isin(orgs)].copy(deep=True)
```

```python
#CTDEEP year-round stations
year_round = ["A4", "B3", "C1", "C2", "D3", "E1", "09", "15", "F2", "F3", "H2", "H4", "H6", "I2", "J2", "K2", "M3"]
year_round = ["CT_DEP01_WQX-" + x for x in year_round]
year_round
```

```python
#Marking nas properly
trend_data["ResultMeasureValue"] = pd.to_numeric(trend_data["ResultMeasureValue"], errors='coerce')
```

```python
#Properly setting datatypes
trend_data["ResultMeasureValue"]=pd.to_numeric(trend_data["ResultMeasureValue"])
trend_data["ResultMeasureValue"].dtype
```

```python
#Formatting dates
trend_data["ActivityStartDate"]=pd.to_datetime(trend_data["ActivityStartDate"])
```

```python
#Getting usable stations for each org
usable_st={}
for org in orgs:
    IEC_counts =agg_data.loc[agg_data["OrganizationIdentifier"]==org].groupby(["year", "MonitoringLocationIdentifier"])["ResultMeasureValue"].count().unstack(0)
    print(IEC_counts.head(30))
    usable=IEC_counts.dropna(how="any")
    print(list(usable.index))
    usable_st[org]=usable.index
```

```python
#Station Names
st_names=agg_data[["MonitoringLocationIdentifier", "MonitoringLocationName"]].drop_duplicates("MonitoringLocationIdentifier")
st_names=dict(zip(st_names["MonitoringLocationIdentifier"], st_names["MonitoringLocationName"]))
print(st_names)
```

```python
plt.rcParams["figure.figsize"]=(21, 14)
```

# Getting time of year of sampling

```python
trend_data["day_of_year"]=trend_data["ActivityStartDate"].dt.dayofyear
```

```python
#Dealing with time outlier
print(trend_data["ActivityStartDate"].min())
```

```python
plt.rcParams["figure.figsize"]=(21, 14)
```

```python
#Remember to Group By MonitoringLocationIdentifier because station names can be duplicated

for org in orgs:
    locs = list(trend_data.loc[trend_data["OrganizationIdentifier"]==org, "MonitoringLocationIdentifier"].drop_duplicates())
    fig, ax = plt.subplots()
    print(len(locs))
    for loc in locs:
        x=trend_data.loc[(trend_data["OrganizationIdentifier"]==org) & (trend_data["MonitoringLocationIdentifier"]==loc), "day_of_year"].values
        y=trend_data.loc[(trend_data["OrganizationIdentifier"]==org) & (trend_data["MonitoringLocationIdentifier"]==loc), "ResultMeasureValue"].values
        plt.scatter(x=x, y=y)
    plt.show()
```

```python
#Getting std on a daily timescale for each org

for org in orgs:
    fig, ax=plt.subplots()
    stddev=trend_data.loc[trend_data["OrganizationIdentifier"]==org].groupby("ActivityStartDate")["ResultMeasureValue"].std()
    #stddev.reset_index(inplace=True)
    print(stddev.head())
    print(stddev.mean())
    plt.scatter(x=stddev, y=[1]*len(stddev))
    plt.scatter(x=stddev.mean(), y=1, s=100)
    ax.set_title(org + " Daily Standard Deviations")
    plt.show()
```

```python
#Pre-cursor to outlier analysis
import datetime
outlier_orgs={}

for org in orgs:
    outlier_orgs[org]={}
    outlier_dt=outlier_orgs[org]
    org_data=trend_data.loc[(trend_data["OrganizationIdentifier"]==org)].set_index("ActivityStartDate", drop=False)
    org_data["ActivityDepthHeightMeasure/MeasureValue"]=-1*org_data["ActivityDepthHeightMeasure/MeasureValue"]
    fig, ax = plt.subplots()
    print(len(locs))
    for date in pd.unique(org_data.index):
        if (org_data.loc[date, "ResultMeasureValue"].min()<7) | (org_data.loc[date, "ResultMeasureValue"].min()>10):
            outlier_dt[date]=org_data.loc[date]
            x=org_data.loc[date, "ActivityStartDate"]
            y=org_data.loc[date, "ResultMeasureValue"]
            plt.scatter(x=x, y=y, label=str(date)[:10])
    plt.legend(title="Date", loc='upper left')
    plt.show()
```

```python
for org in orgs:   
    stddf=pd.Series()
    for date, outlier in outlier_orgs[org].items():
        #summary=outlier["ResultMeasureValue"].describe()
        stddf.loc[len(stddf)]=outlier["ResultMeasureValue"].std()
        #print(summary["50%"]-1.5*(summary["75%"]-summary["25%"]))
        #fig, ax = plt.subplots()
        #outlier.boxplot("ResultMeasureValue", vert=False)
        #ax.set_title(org + " " + str(date)[:9])
        #plt.show()
    print(org + ":")
    print(stddf.mean())
```

```python
for org in orgs:   
    if org=="CT_DEP01_WQX":
        for date, outlier in outlier_orgs[org].items():
            print(date)
            fig, ax =plt.subplots()
            for st in pd.unique(outlier["MonitoringLocationIdentifier"]):
                x=outlier.loc[outlier["MonitoringLocationIdentifier"]==st]["ResultMeasureValue"]
                y=outlier.loc[outlier["MonitoringLocationIdentifier"]==st]["ActivityDepthHeightMeasure/MeasureValue"]
                plt.scatter(x, y, label=st_names[st])
                plt.legend(title="Location", loc='upper left')
                name=org + " " + str(date)[:10]
                ax.set_title(name)
            plt.savefig("Charts 09-16/" + name + ".png")
            plt.show()
    else:
        pass
```

```python
help(pd.DataFrame.set_index)
```

# Outlier Analysis

```python
trend_data.loc[pd.to_numeric(trend_data["ResultMeasureValue"], errors="coerce").isna()]
```

```python
#Variance in each year
innervar = trend_data.groupby(["year", "OrganizationIdentifier"])["ResultMeasureValue"].var()
innervar.unstack(0).head(15)
```

```python
bad_cruises= trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (trend_data["ResultMeasureValue"]>10), "cruise name"].drop_duplicates()
trend_data.drop(trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (trend_data["cruise name"].isin(bad_cruises))].index, inplace=True)
```

```python
#Variance in each year
innervar = trend_data.groupby(["year", "OrganizationIdentifier"])["ResultMeasureValue"].var()
innervar.unstack(0).head(15)
```

```python
trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & ((trend_data["year"]==2017)), ["ResultMeasureValue", "ActivityStartDate"]].sort_values("ResultMeasureValue").tail(100)

```

```python
innervar = trend_data.groupby(["year", "OrganizationIdentifier"])["ResultMeasureValue"].var()
innervar.unstack(0).head(15)
```

```python
#Dealing with small outliers
trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (trend_data["year"]==2010)].sort_values("ResultMeasureValue", ascending=True)
trend_data.drop(trend_data.loc[trend_data["ResultMeasureValue"]<= 4].index, inplace=True)
```

# Yearly Trends (not so useful)

```python
plt.rcParams["figure.figsize"]=(21, 14)
```

```python
for org in orgs:
    fig, ax = plt.subplots()
    org_data = trend_data.loc[(trend_data["OrganizationIdentifier"]==org) & (trend_data["MonitoringLocationIdentifier"].isin(usable_st[org]))]
    means = org_data.groupby(["MonitoringLocationIdentifier", "year"])["ResultMeasureValue"].mean()
    means.unstack(0).plot(ax=ax)
    plt.show()
```

# More granular trends

```python

for org in orgs:
    fig, ax = plt.subplots()
    for st in usable_st[org]:
        org_data=trend_data.set_index("ActivityStartDate")
        org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
    plt.show()
```

```python
#pH below 7

fig, ax = plt.subplots()
ax.set_title("CTDEEP Stations- Year Round")
for st in year_round:
    org_data=trend_data.set_index("ActivityStartDate")
    org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
    
org_data=trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (trend_data["MonitoringLocationIdentifier"]).isin(year_round)]

for year in pd.unique(trend_data["year"]):
    for st in pd.unique(org_data.loc[org_data["year"]==year, "MonitoringLocationIdentifier"]):
        smallest = org_data.loc[(org_data["MonitoringLocationIdentifier"]==st) & (org_data["year"]==year)].sort_values("ResultMeasureValue", ascending=True).iloc[0]
        if (smallest["ResultMeasureValue"] < 7):
            plt.plot(smallest["ActivityStartDate"], smallest["ResultMeasureValue"], marker="o", markersize=10, color="black")
            ax.annotate("  " + smallest["ActivityStartDate"].month_name() + " " +str(smallest["ActivityStartDate"].day), (smallest["ActivityStartDate"], smallest["ResultMeasureValue"]))
plt.show()
```

```python
#Seperating CTDEEP Year-Round and other stations
fig, ax = plt.subplots()
ax.set_title("CTDEEP Stations- Year Round")
for st in year_round:
    org_data=trend_data.set_index("ActivityStartDate")
    org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
for year in pd.unique(trend_data["year"]):
    smallest = trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (trend_data["year"]==year)].sort_values("ResultMeasureValue", ascending=True).iloc[0]
    plt.plot(smallest["ActivityStartDate"], smallest["ResultMeasureValue"], marker="o", markersize=10, color="black")
    ax.annotate("  " + smallest["ActivityStartDate"].month_name() + " " +str(smallest["ActivityStartDate"].day) + ", " + str(smallest["ActivityDepthHeightMeasure/MeasureValue"]) +", " + smallest["MonitoringLocationName"], (smallest["ActivityStartDate"], smallest["ResultMeasureValue"]))
plt.savefig("Charts-09-07/CTDEEP_year_round.png")
plt.show()

fig, ax = plt.subplots()
ax.set_title("CTDEEP Stations- Seasonal")
for st in list(set(usable_st["CT_DEP01_WQX"]) - set(year_round)):
    org_data=trend_data.set_index("ActivityStartDate")
    org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
plt.savefig("Charts-09-07/CTDEEP_seasonal.png")
plt.show()
```

# Trends by the Season

```python
seasons = dict(zip(np.arange(1,13), ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]))
print(seasons)
for org in orgs:
    for season in np.arange(1,13):
        month_data=trend_data.loc[(trend_data["OrganizationIdentifier"]==org) & (trend_data["ActivityStartDate"].dt.month==season)].copy(deep=True)
        print(season)
        counts = month_data.groupby(["year", "MonitoringLocationIdentifier"])["ResultMeasureValue"].count().unstack(0)
        counts.replace(0, np.nan, inplace=True)
        counts = (1-counts.isna())
        counts = counts.sum(axis=1)
        #print(counts)
        usable = counts.loc[counts>10].index
        
        month_data=month_data.groupby(["MonitoringLocationIdentifier", "year"]).mean()
        month_data.reset_index(inplace=True)
        #print(month_data)
        if len(usable)>0:
            fig, ax = plt.subplots()
            ax.set_title(seasons[season] + " " + org)
            for st in usable:
                plt.plot(month_data.loc[month_data["MonitoringLocationIdentifier"]==st, "year"], month_data.loc[month_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"], label=st.lstrip(org + "-"))
            plt.legend(title="Station", loc='upper right')
            plt.savefig("Charts-09-07/" + org + "/" + seasons[season] + " " + org +".png", bbox_inches='tight')            
            plt.show()
        
```

```python

```

```python

```

# Rolling Averages

```python
#Setting index to date
trend_data.set_index("ActivityStartDate", inplace=True)
```

```python
plt.rcParams["figure.figsize"]=(40, 14)
```

```python
#Remember to Group By MonitoringLocationIdentifier because station names can be duplicated

for org in orgs:
    fig, ax = plt.subplots()
    trend_data.loc[trend_data["OrganizationIdentifier"]==org].groupby("MonitoringLocationIdentifier")["ResultMeasureValue"].plot(ax=ax)
    plt.show()
```

```python
trend_data["ResultMeasureValue"]
```

```python
#Smoothing to calculate rolling mean and rolling stddev
from pandas.tseries.offsets import DateOffset

for org in orgs:
    org_data=trend_data.loc[(trend_data["OrganizationIdentifier"]==org) & (trend_data["MonitoringLocationIdentifier"].isin(usable_st[org]))].copy(deep=True)
    #print(usable_st[org])
    #print(len(org_data))
    org_data.sort_values(by="ActivityStartDate", axis=0, inplace=True)
    
    rolmean = org_data.groupby("MonitoringLocationIdentifier")["ResultMeasureValue"].rolling("365d").mean(center=True)
    rolstd = org_data.groupby("MonitoringLocationIdentifier")["ResultMeasureValue"].rolling("365d").std(center=True)
    
    #Testing for accuracy
    print(rolmean.loc["31ISC2RS_WQX-8-403"])
    print(org_data.loc[org_data["MonitoringLocationIdentifier"]=="31ISC2RS_WQX-8-403", "ResultMeasureValue"])
    print(org_data.index.min())
    print(org_data.index.max())
    
    #rolmean=pd.DataFrame(rolmean).reset_index()
    #print(rolmean["ActivityStartDate"].max)
    
    rolmean=pd.DataFrame(rolmean)
    rolmean.reset_index(inplace=True)
    rolmean.set_index("ActivityStartDate", inplace=True)
    rolmean.groupby("MonitoringLocationIdentifier")["ResultMeasureValue"].plot(legend=True)
    plt.xticks(rotation=45)
    #print(plt.xlim)
    left, right = plt.xlim()
    print(left)
    print(right)
    plt.show()
    
    #rolstd.groupby(level=0).plot(legend=True)
    #plt.show()
```

```python
rolmean=pd.DataFrame(rolmean).reset_index()
print(rolmean)
rolmean.loc[rolmean["MonitoringLocationIdentifier"]=="31ISC2RS_WQX-9-413", "ActivityStartDate"].drop_duplicates()
```

```python
help(pd.DataFrame.sort_values)
```

```python
help(pd.to_datetime)
```

```python

```
