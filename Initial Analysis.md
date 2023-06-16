```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```python
import os
import scipy.fft as fft
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.ndimage import median_filter
```

```python
# #Importing custom idw weighting function
# os.chdir('../Data Visualization and Analytics Challenge')
# from Data import inverse_distance_weighter as idw
# os.chdir('../pH_analysis ')
```

```python
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
```

# Loading the Data

```python
agg_data=pd.read_csv("cleaned_aggregated_pH_data.csv", index_col=0)
agg_data.head()
```

# Fusing Basins to Data

```python
stations=pd.read_csv("../OneDrive - Environmental Protection Agency (EPA)/Downloads/Basin_Labelled_Stations.csv", index_col=[0])
stations
```

```python
agg_data=agg_data.merge(stations[["MonitoringLocationIdentifier", "loc", "area"]], how="left", on="MonitoringLocationIdentifier")
agg_data
```

```python
#Making sure all stations (except previous missing) get a loc/area
agg_data.loc[agg_data["loc"].isna(), "MonitoringLocationIdentifier"].drop_duplicates()
```

# Initial Manipulation

```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% ! important; }<style>"))
```

```python
pd.set_option("display.precision", 2)
```

```python
counted=agg_data.groupby(["year", "OrganizationFormalName"])["ResultMeasureValue"].count()
counted.unstack(0).head(15)
```

```python
#Sampling Frequency
counted=agg_data.drop_duplicates(["ActivityStartDate", "MonitoringLocationIdentifier"]).groupby(["year", "OrganizationFormalName", "MonitoringLocationIdentifier"])["ResultMeasureValue"].count()
freqs=counted.unstack(0)
freqs.reset_index(inplace=True)
mean_freqs=freqs.groupby("OrganizationFormalName").mean()
mean_freqs=mean_freqs.loc[["Connecticut Department Of Energy And Environmental Protection", "Interstate Environmental Commission", "USGS Connecticut Water Science Center", "USGS New York Water Science Center"]]
mean_freqs.fillna(0, inplace=True)
mean_freqs
```

```python
#Selecting organizations with LIS presence
orgs=["31ISC2RS_WQX", "CT_DEP01_WQX", "USGS-CT", "USGS-NY"]
trend_data=agg_data.loc[agg_data["OrganizationIdentifier"].isin(orgs)].copy(deep=True)
```

```python
all_stations=pd.unique(trend_data.loc[trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX", "MonitoringLocationIdentifier"])
print(len(all_stations))
all_stations
```

```python
all_stations=pd.unique(trend_data.loc[trend_data["OrganizationIdentifier"]=="31ISC2RS_WQX", "MonitoringLocationIdentifier"])
print(len(all_stations))
all_stations
```

```python
#CTDEEP year-round stations
year_round = ["A4", "B3", "C1", "C2", "D3", "E1", "09", "15", "F2", "F3", "H2", "H4", "H6", "I2", "J2", "K2", "M3"]
year_round = ["CT_DEP01_WQX-" + x for x in year_round]
print(len(year_round))
year_round
```

```python
#Year-Round Frequency
#Sampling Frequency
counted=agg_data.drop_duplicates(["ActivityStartDate", "MonitoringLocationIdentifier"]).groupby(["year", "MonitoringLocationIdentifier", "OrganizationFormalName"])["ResultMeasureValue"].count()
freqs=counted.unstack(0)
freqs=freqs.loc[year_round]
freqs.reset_index(inplace=True)
mean_freqs=freqs.groupby("OrganizationFormalName").mean()
mean_freqs.fillna(0, inplace=True)
mean_freqs
```

```python
#Marking nas properly
trend_data["ResultMeasureValue"] = pd.to_numeric(trend_data["ResultMeasureValue"], errors='coerce')
trend_data["ResultMeasureValue"].dtype
```

```python
#Formatting dates
trend_data["ActivityStartDate"]=pd.to_datetime(trend_data["ActivityStartDate"])
```

```python
#Getting usable stations for each org (stations with a complete time series)
usable_st={}
for org in orgs:
    IEC_counts =agg_data.loc[agg_data["OrganizationIdentifier"]==org].groupby(["year", "MonitoringLocationIdentifier"])["ResultMeasureValue"].count().unstack(0)
    print(IEC_counts.head(30))
    usable=IEC_counts.dropna(how="any")
    print(list(usable.index))
    usable_st[org]=usable.index
```

```python
#Station Names Dict
st_names=agg_data[["MonitoringLocationIdentifier", "MonitoringLocationName"]].drop_duplicates("MonitoringLocationIdentifier")
st_names=dict(zip(st_names["MonitoringLocationIdentifier"], st_names["MonitoringLocationName"]))
print(st_names)
```

```python
#Organization Names Dict
print(pd.unique(agg_data["OrganizationFormalName"]))
print(pd.unique(agg_data["OrganizationIdentifier"]))
org_names=dict(zip(pd.unique(agg_data["OrganizationIdentifier"]), pd.unique(agg_data["OrganizationFormalName"])))

#Manually changing some for brevity
org_names["CT_DEP01_WQX"]="CTDEEP"
org_names["31ISC2RS_WQX"]="IEC"
org_names["USGS-CT"]="USGS-CT"
org_names["USGS-NY"]="USGS-NY"
org_names
```

```python
#Dealing with nas
print(len(trend_data.loc[trend_data["ResultMeasureValue"].isna()]))
trend_data.dropna(subset=["ResultMeasureValue"], inplace=True)
trend_data.loc[pd.to_numeric(trend_data["ResultMeasureValue"], errors="coerce").isna()]
```

```python
#Setting size of figures
plt.rcParams["figure.figsize"]=(21, 10)
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
#Remember to Group By MonitoringLocationIdentifier because station names can be duplicated

for org in orgs:
    for basin in np.arange(1, 6):
        working=trend_data.loc[trend_data["area"]==basin]
        locs = list(working.loc[working["OrganizationIdentifier"]==org, "MonitoringLocationIdentifier"].drop_duplicates())
        print(len(locs))
        if len(working.loc[working["OrganizationIdentifier"]==org])>0:
            fig, ax = plt.subplots()
            for loc in locs:
                x=working.loc[(working["OrganizationIdentifier"]==org) & (working["MonitoringLocationIdentifier"]==loc), "day_of_year"].values
                y=working.loc[(working["OrganizationIdentifier"]==org) & (working["MonitoringLocationIdentifier"]==loc), "ResultMeasureValue"].values
                plt.scatter(x=x, y=y)
            ax.set_title("Basin " + str(basin) + " " + str(org_names[org]))
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
    if len(org_data)>0:
        fig, ax = plt.subplots()
        print(len(locs))
        for date in pd.unique(org_data.index):
            if (org_data.loc[date, "ResultMeasureValue"].max()>10):
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
    print(org_names[org] + ":")
    print(stddf.mean())
```

```python
# Depth analysis (not basin-specific for now)
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
                name=org_names[org] + " " + str(date)[:10]
                ax.set_title(name)
            #plt.savefig("Charts 09-16/" + name + ".png")
            plt.show()
    else:
        pass
```

```python
#Depth Analysis by Date (for finding errors)

#param
cruise = "WQSEP18"

#graphing
working=trend_data.loc[trend_data["cruise name"]==cruise]
for st in [s for s in pd.unique(working["MonitoringLocationIdentifier"]) if (((st_names[s][0])!="0") & ((st_names[s][0])!="1"))]:
    x=working.loc[working["MonitoringLocationIdentifier"]==st]["ResultMeasureValue"]
    y=working.loc[working["MonitoringLocationIdentifier"]==st]["ActivityDepthHeightMeasure/MeasureValue"]
    plt.scatter(x, y, label=st_names[st])
    plt.legend(title="Location", loc='upper left')
    name=org_names[org] + " " + str(date)[:10]
    ax.set_title(name)
#plt.savefig("Charts 09-16/" + name + ".png")
plt.show()
```

```python
# Trying to find outlier cruises
bad_cruises= trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (trend_data["ResultMeasureValue"]>10), "cruise name"].drop_duplicates()
bad_cruises
```

```python
for cruise in bad_cruises:
    print(trend_data.loc[trend_data["cruise name"]==cruise].head())
    print(trend_data.loc[trend_data["cruise name"]==cruise].mean())
```

# Getting rid of bad cruises

```python
#Reading in notes on ph_errors
ph_notes=pd.read_csv("Notes_on_pH_malfunction.csv")
```

```python
to_drop=pd.unique(ph_notes["Cruise"])
to_drop
```

```python
filtered=trend_data.loc[~trend_data["cruise name"].isin(to_drop)]
print(len(trend_data))
print(len(filtered))
```

```python
#Seeing which cruises still have very high phs
still_iffy=filtered.loc[(filtered["OrganizationIdentifier"]=="CT_DEP01_WQX") & (filtered["ResultMeasureValue"]>10), "cruise name"].drop_duplicates()
still_iffy
```

```python
#Depth Analysis by Date (for finding errors)

for cruise in list(still_iffy):

    #graphing
    working=trend_data.loc[trend_data["cruise name"]==cruise]
    for st in year_round:
        x=working.loc[working["MonitoringLocationIdentifier"]==st]["ResultMeasureValue"]
        y=working.loc[working["MonitoringLocationIdentifier"]==st]["ActivityDepthHeightMeasure/MeasureValue"]
        plt.scatter(x, y, label=st_names[st])
        plt.legend(title="Location", loc='upper left')
        name=org_names[org] + " " + str(date)[:10]
        ax.set_title(name)
    #plt.savefig("Charts 09-16/" + name + ".png")
    plt.title(cruise)
    plt.show()
```

```python
#Dropping the three unconfirmed from that time period after referencing field notes
to_drop_ii=["WQNOV18", "WQDEC18", "WQJAN19", "WQNOV17", "WQJUN21"]
print(len(filtered))
filtered=filtered.loc[~filtered["cruise name"].isin(to_drop_ii)].copy(deep=True)
```

```python
#new length
print(len(filtered))
```

```python
#Seeing which cruises which have very low phs
still_iffy=filtered.loc[(filtered["OrganizationIdentifier"]=="CT_DEP01_WQX") & (filtered["ResultMeasureValue"]>10), "cruise name"].drop_duplicates()
still_iffy
```

```python
#Depth Analysis by Date (for finding errors)

for cruise in list(still_iffy):

    #graphing
    working=trend_data.loc[trend_data["cruise name"]==cruise]
    for st in year_round:
        x=working.loc[working["MonitoringLocationIdentifier"]==st]["ResultMeasureValue"]
        y=working.loc[working["MonitoringLocationIdentifier"]==st]["ActivityDepthHeightMeasure/MeasureValue"]
        plt.scatter(x, y, label=st_names[st])
        plt.legend(title="Location", loc='upper left')
        name=org_names[org] + " " + str(date)[:10]
        ax.set_title(name)
    #plt.savefig("Charts 09-16/" + name + ".png")
    plt.title(cruise)
    plt.show()
```

# Autocorrelation and Despiking Year Round Data

```python
#Converting depths to numeric
filtered["ActivityDepthHeightMeasure/MeasureValue"]=pd.to_numeric(filtered["ActivityDepthHeightMeasure/MeasureValue"], errors="coerce")
filtered.dropna(subset="ActivityDepthHeightMeasure/MeasureValue", axis=0, inplace=True)
print(len(filtered.dropna(subset="ActivityDepthHeightMeasure/MeasureValue", axis=0)))
print(len(filtered.dropna(subset="ResultMeasureValue", axis=0)))
print(len(filtered.dropna(subset="MonitoringLocationIdentifier", axis=0)))
```

```python
#Repeating the procedure from temp time series
ph_dict={}
for station in pd.unique(year_round):
    working=filtered.loc[filtered["MonitoringLocationIdentifier"]==station].copy(deep=True)
    working["Date"]=pd.to_datetime(working["ActivityStartDate"])
    working["Year"]=working["Date"].dt.year
    
    #saving
    ph_dict[station]=working
    
```

```python
sum([len(df) for df in ph_dict.values()])
```

```python
#Seperating by segment of the depth profile
ph_depth_dict={}
for i in np.arange(int(filtered["ActivityDepthHeightMeasure/MeasureValue"].max())+1):
    ph_depth_dict[i]={}
    subdict=ph_depth_dict[i]
    for index, df in ph_dict.items():
        working=df.copy(deep=True)
        working.dropna(subset="ActivityDepthHeightMeasure/MeasureValue", inplace=True)
        working=working.loc[(working["ActivityDepthHeightMeasure/MeasureValue"]>=i) & (working["ActivityDepthHeightMeasure/MeasureValue"]<(i+1))]
        subdict[index]=working
```

## Autocorrelation (skip for now because of uneven intervals)

```python
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
```

```python
#Trial doing autocorrelation by depth alone (all stations in one corrrelogram)
for depth in [d for d in pd.unique(filtered["ActivityDepthHeightMeasure/MeasureValue"]) if ((d>=2) & (d<=3))]:
    working=filtered.loc[filtered["ActivityDepthHeightMeasure/MeasureValue"]==depth].copy(deep=True)
    working.dropna(subset=["ResultMeasureValue"], axis=0, inplace=True)
    print(pd.unique(working["ActivityStartDate"]))
    print(len(working))
    plot_acf(working["ResultMeasureValue"], lags=np.linspace(0, int(len(working)/2), num=20, endpoint=True), missing="conservative")
    plt.title(depth)
    plt.show()
```

```python
ph_dict["CT_DEP01_WQX-C2"]
```

```python
for index, df in ph_dict.items():
    plt.scatter(df["Date"], df[""ActivityDepthHeightMeasure/MeasureValue""])
    plt.xticks(rotation=90)
    plt.show()
```

```python
#Autocorrelation
for station in pd.unique(data["Station ID"]):
    working=data.loc[data["Station ID"]==station]
    timespan=max(working["Date Time (GMT-04:00)"].dt.year)-min(working["Date Time (GMT-04:00)"].dt.year)+1
    print(timespan)
    working.set_index("Date Time (GMT-04:00)", inplace=True)
    plot_acf(working["Temperature (C)"], lags=np.linspace(0, len(working)-1, num=50, endpoint=True), missing="conservative")
    locs, labels= plt.xticks()
    months=int(locs[-1]/(96*30.5))
    plt.xticks(np.linspace(0, months*96*30.5, months+1), np.linspace(0, months, months+1))
    plt.xticks(rotation=90)
    plt.show()
```

```python
print(ph_depth_dict.keys())
```

```python
#Graphing params
drange=2
```

```python
#Graphing
for index, df in ph_depth_dict[drange].items():
    plt.scatter(df["Date"], df["ResultMeasureValue"])
    plt.xticks(rotation=90)
    plt.title(index + " at depth " + str(i) + " to " + str(i+1) + " m")
    plt.show()
```

## Despiking

```python
#Threshold for despiking (the endpoint threshold should be less)
window=5
thresh=.5
```

```python
#Despiking function for non-endpoints
def despike(df, threshold, window_size):
    filtered=median_filter(df.values, size=window_size)
    despiked=np.where(np.abs(filtered-df.values)>=threshold, filtered, df.values)
    return(pd.DataFrame(despiked, index=df.index))
```

```python
#Getting depths
depths = list(ph_depth_dict.keys()).copy()

```

```python
#Despiking by depth range
post_depth_dict={}
for depth in depths:
    post_depth_dict[depth]={}
    for index, df in ph_depth_dict[depth].items():
        #Ensuring proper order
        working=df.sort_values(["Date", "ActivityDepthHeightMeasure/MeasureValue"])

        #Setting index as needed for function
        working.set_index(["MonitoringLocationIdentifier", "Date", "year"], inplace=True, drop=True)

        #Despiking
        working["ResultMeasureValue"]=despike(working["ResultMeasureValue"], thresh, window)
        working.reset_index(inplace=True)
        print(working["ResultMeasureValue"].dropna())
        post_depth_dict[depth][index]=working
        
```

```python
print(post_depth_dict.keys())
```

```python
#Testing param
drange=0
```

```python
#Testing threshold
for index, df in ph_depth_dict[drange].items():
    working=df.sort_values(["Date", "ActivityDepthHeightMeasure/MeasureValue"])
    working.set_index("Date", inplace=True)
    #print(working.loc[working["Year"]==2019, ["ResultMeasureValue", "ActivityDepthHeightMeasure/MeasureValue"]])
    med_filtered=pd.DataFrame(median_filter(working["ResultMeasureValue"].values, size=5), index=working.index, columns=["ResultMeasureValue"])
    error=pd.DataFrame(med_filtered["ResultMeasureValue"].values-working["ResultMeasureValue"].values, index=working.index)
    working.reset_index(inplace=True)
    error.reset_index(inplace=True, drop=True)
    print(error.shape)
    print(index)
    for idx, datapoint in error.loc[np.abs(error.values)>.5].iterrows():
        if index=="CT_DEP01_WQX-F2":
            print(datapoint)
            print(working.loc[idx-2:idx+2, ["cruise name", "ActivityStartDate","ActivityStartTime/Time", "ResultMeasureValue", "ActivityDepthHeightMeasure/MeasureValue"]])
```

```python
#Testing after despike
for index, df in post_depth_dict[drange].items():
    working=df.sort_values(["Date", "ActivityDepthHeightMeasure/MeasureValue"])
    working.set_index("Date", inplace=True)
    #print(working.loc[working["Year"]==2019, ["ResultMeasureValue", "ActivityDepthHeightMeasure/MeasureValue"]])
    med_filtered=pd.DataFrame(median_filter(working["ResultMeasureValue"].values, size=5), index=working.index, columns=["ResultMeasureValue"])
    error=pd.DataFrame(med_filtered["ResultMeasureValue"].values-working["ResultMeasureValue"].values, index=working.index)
    working.reset_index(inplace=True)
    error.reset_index(inplace=True, drop=True)
    print(error.shape)
    print(index)
    for idx, datapoint in error.loc[np.abs(error.values)>.5].iterrows():
        print(datapoint)
        print(working.loc[idx-2:idx+2, ["MonitoringLocationIdentifier", "ActivityStartDate","ActivityStartTime/Time", "ResultMeasureValue", "ActivityDepthHeightMeasure/MeasureValue"]])
```

Based on the above testing, the despiking threshold seems right, by keeping the window at 5 and having samples ordered by depth, we ensure that we can only eliminate a sample if it represents a dramatic departure from the previous/next measurement date, the rest of this date's depth profile within the meter, or both.

```python
#Graphing param
drange=2
```

```python
#Graphing
for index, df in post_depth_dict[drange].items():
    working=df.copy(deep=True)
    working.reset_index(inplace=True)
    plt.scatter(working["Date"], working["ResultMeasureValue"])
    plt.xticks(rotation=90)
    plt.title(index + " at chosen depth")
    plt.show()
```

# Outlier Analysis

```python
depth_agg={}
for depth in depths:
    dfs = post_depth_dict[depth].values()
    depth_agg[depth]=pd.concat(list(post_depth_dict[depth].values()))

despiked_agg=pd.concat(list(depth_agg.values()), axis=0)
print(len(despiked_agg))
despiked_agg
```

## Post-despiked Depth Profile

```python
#Seeing which cruises which have very low phs
low=despiked_agg.loc[(despiked_agg["OrganizationIdentifier"]=="CT_DEP01_WQX") & (despiked_agg["ResultMeasureValue"]<6), "cruise name"].drop_duplicates()
low
```

```python
#Depth Analysis by Date (for finding errors)

for cruise in list(low):

    #graphing
    working=despiked_agg.loc[despiked_agg["cruise name"]==cruise]
    for st in year_round:
        x=working.loc[working["MonitoringLocationIdentifier"]==st]["ResultMeasureValue"]
        y=-working.loc[working["MonitoringLocationIdentifier"]==st]["ActivityDepthHeightMeasure/MeasureValue"]
        plt.scatter(x, y, label=st_names[st])
        name=org_names[org] + " " + str(date)[:10]
        
    plt.legend(loc="upper right", fontsize="xx-large", bbox_to_anchor=(1.1, 1))
    ax=plt.gca()
    ax.set_xlabel("pH", fontsize="xx-large")
    ax.set_ylabel("Depth", fontsize="xx-large")
    ax.set_title("Cruise: " + cruise, fontsize="xx-large")
    plt.savefig("../OneDrive - Environmental Protection Agency (EPA)/Downloads/WQMWG Presentation/CTDEEP_" + cruise + "_depth_profile.png", bbox_inches='tight')
    plt.show()
```

```python
#Outputting despiked_agg to be used later
despiked_agg.to_csv("../Data Visualization and Analytics Challenge/Data/processed_ph_1.24.csv")
```

```python
print(len(despiked_agg))
```

```python
despiked_agg.reset_index(inplace=True)
despiked_agg.head()
```

```python
despiked_agg.loc[pd.to_numeric(despiked_agg["ResultMeasureValue"], errors="coerce").isna()]
```

```python
trend_data["Year"]=trend_data["ActivityStartDate"].dt.year
```

```python
#Variance in each year
innervar = trend_data.loc[trend_data["MonitoringLocationIdentifier"].isin(year_round)].groupby(["Year", "OrganizationIdentifier"])["ResultMeasureValue"].var()
print(innervar.unstack(0).head(15))
innervar_post = despiked_agg.groupby(["Year", "OrganizationIdentifier"])["ResultMeasureValue"].var()
print(innervar_post.unstack(0).head(15))
```

```python
despiked_agg.loc[(despiked_agg["OrganizationIdentifier"]=="CT_DEP01_WQX") & ((despiked_agg["year"]==2017)), ["ResultMeasureValue", "ActivityStartDate"]].sort_values("ResultMeasureValue").tail(100)

```

```python
#Dealing with small outliers
despiked_agg.loc[despiked_agg["ResultMeasureValue"]<6]
```

```python
#Seeing Near Small Outliers
for idx, datapoint in despiked_agg.loc[despiked_agg["ResultMeasureValue"]<6.5].iterrows():
    print(datapoint["ResultMeasureValue"])
    print(despiked_agg.loc[idx-4:idx+4,"ResultMeasureValue"].median())
    print(despiked_agg.loc[idx-4:idx+4, ["MonitoringLocationIdentifier","ActivityStartTime/Time", "ResultMeasureValue", "ActivityDepthHeightMeasure/MeasureValue"]])
```

```python
# #Seeing Near Small Outliers with high depth
# for idx, datapoint in despiked_agg.loc[(despiked_agg["ResultMeasureValue"]<6.5) & (despiked_agg["Depth"]>2)].iterrows():
#     print(datapoint["ResultMeasureValue"])
#     print(despiked_agg.loc[idx-4:idx+4,"ResultMeasureValue"].median())
#     print(despiked_agg.loc[idx-4:idx+4, ["MonitoringLocationIdentifier","ActivityStartTime/Time", "Date", "ResultMeasureValue", "Depth"]])
```

# Yearly Trends (not so useful)

```python
plt.rcParams["figure.figsize"]=(21, 14)
```

```python
for org in orgs:
    for basin in np.arange(1, 6):
        working=trend_data.loc[trend_data["area"]==basin]
        org_data = working.loc[(working["OrganizationIdentifier"]==org) & (working["MonitoringLocationIdentifier"].isin(usable_st[org]))]
        if len(org_data)>0:
            fig, ax = plt.subplots()
            means = org_data.groupby(["MonitoringLocationIdentifier", "year"])["ResultMeasureValue"].mean()
            means.unstack(0).plot(ax=ax)
            ax.set_title("Basin " + str(basin) + " " + str(org_names[org]))
            plt.show()
```

```python
#Same analysis as above for CTDEEP but with Year-Round stations separated
org="CT_DEP01_WQX"
for basin in np.arange(1, 6):
    
        working=trend_data.loc[trend_data["area"]==basin]
        org_data = working.loc[(working["OrganizationIdentifier"]==org) & (working["MonitoringLocationIdentifier"].isin(usable_st[org])) & (~working["MonitoringLocationIdentifier"].isin(year_round))]
        if len(org_data)>0:
            fig, ax = plt.subplots()
            means = org_data.groupby(["MonitoringLocationIdentifier", "year"])["ResultMeasureValue"].mean()
            means.unstack(0).plot(ax=ax)
            ax.set_title("Basin " + str(basin) + " " + str(org_names[org])+ " Seasonal")
            plt.show()
        org_data = working.loc[(working["OrganizationIdentifier"]==org) & (working["MonitoringLocationIdentifier"].isin(usable_st[org])) & (working["MonitoringLocationIdentifier"].isin(year_round))]
        if len(org_data)>0:
            fig, ax = plt.subplots()
            means = org_data.groupby(["MonitoringLocationIdentifier", "year"])["ResultMeasureValue"].mean()
            means.unstack(0).plot(ax=ax)
            ax.set_title("Basin " + str(basin) + " " + str(org_names[org]) + " Year Round")
            plt.show()
```

# More granular trends


## Overall

```python
for basin in np.arange(1, 6):
    working=despiked_agg.loc[despiked_agg["area"]==basin]
    for org in orgs:
        if len(working)>0:
            fig, ax = plt.subplots()
            for st in usable_st[org]:
                org_data=working.set_index("ActivityStartDate")
                org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
            ax.set_title("Basin " + str(basin) + " " + str(org_names[org]))
            plt.show()
```

```python
#Plotting Year-Round Data
for basin in np.arange(1, 6):
    working=despiked_agg.loc[despiked_agg["area"]==basin]
    fig, ax = plt.subplots()
    ax.set_title("Basin " + str(basin) + " CTDEEP Stations- Year Round", fontsize="xx-large")
    for st in year_round:
        org_data=working.set_index("ActivityStartDate")
        org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax, label=st_names[st])
    
# #Labeling plots with pH below 7
#     org_data=working.loc[(working["OrganizationIdentifier"]=="CT_DEP01_WQX") & (working["MonitoringLocationIdentifier"]).isin(year_round)]
#     for year in pd.unique(trend_data["year"]):
#         for st in pd.unique(org_data.loc[org_data["year"]==year, "MonitoringLocationIdentifier"]):
#             smallest = org_data.loc[(org_data["MonitoringLocationIdentifier"]==st) & (org_data["year"]==year)].sort_values("ResultMeasureValue", ascending=True).iloc[0]
#             if (smallest["ResultMeasureValue"] < 7):
#                 plt.plot(smallest["ActivityStartDate"], smallest["ResultMeasureValue"], marker="o", markersize=10, color="black")
#                 #ax.annotate("  " + smallest["ActivityStartDate"].month_name() + " " +str(smallest["ActivityStartDate"].day), (smallest["ActivityStartDate"], smallest["ResultMeasureValue"]))
    plt.legend(loc="upper right", fontsize="xx-large", bbox_to_anchor=(1.1, 1))
    ax.set_xlabel("Sampling Date", fontsize="xx-large")
    ax.set_ylabel("pH", fontsize="xx-large")
    plt.savefig("Charts-01-16/CTDEEP_Basin_" + str(basin) + ".png")
    
    plt.show()
```

```python
#Seperating CTDEEP Year-Round and other stations
fig, ax = plt.subplots()
ax.set_title("CTDEEP Stations- Year Round")
for st in year_round:
    org_data=despiked_agg.set_index("ActivityStartDate")
    org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
# for year in pd.unique(trend_data["year"]):
#     smallest = trend_data.loc[(trend_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (trend_data["year"]==year)].sort_values("ResultMeasureValue", ascending=True).iloc[0]
#     plt.plot(smallest["ActivityStartDate"], smallest["ResultMeasureValue"], marker="o", markersize=10, color="black")
#     ax.annotate("  " + smallest["ActivityStartDate"].month_name() + " " +str(smallest["ActivityStartDate"].day) + ", " + str(smallest["ActivityDepthHeightMeasure/MeasureValue"]) +", " + smallest["MonitoringLocationName"], (smallest["ActivityStartDate"], smallest["ResultMeasureValue"]))
plt.savefig("Charts-01-16/CTDEEP_year_round.png")
plt.show()

#Non-year round CTDEEP Stations (Not despiked)
fig, ax = plt.subplots()
ax.set_title("CTDEEP Stations- Seasonal")
for st in list(set(usable_st["CT_DEP01_WQX"]) - set(year_round)):
    org_data=trend_data.set_index("ActivityStartDate")
    org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
plt.savefig("Charts-09-07/CTDEEP_seasonal.png")
plt.show()
```

## By Depth section and basin

```python
#Renaming for Convenience
despiked_agg.rename(columns={"ActivityDepthHeightMeasure/MeasureValue": "Depth"}, inplace=True)
```

```python
#Graphing param
drange=3
```

```python
for basin in np.arange(1, 6):
    working=despiked_agg.loc[(despiked_agg["area"]==basin) & (despiked_agg["Depth"]>=drange) & (despiked_agg["Depth"]<(drange+1))]
    fig, ax = plt.subplots()
    ax.set_title("Basin " + str(basin) + " CTDEEP Stations- Year Round at depth " +"[" + str(drange) + ", " +str(drange + 1) + ") m")
    for st in year_round:
        org_data=working.set_index("ActivityStartDate")
        org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
    
```

## Rolling By Depth 

```python
for basin in np.arange(1, 6):
    working=despiked_agg.loc[(despiked_agg["area"]==basin) & (despiked_agg["Depth"]>=drange) & (despiked_agg["Depth"]<(drange+1))]
    fig, ax = plt.subplots()
    ax.set_title("Basin " + str(basin) + " CTDEEP Stations- Year Round at depth " +"[" + str(drange) + ", " +str(drange + 1) + ") m")
    for st in year_round:
        station_data=working.loc[working["MonitoringLocationIdentifier"]==st, ["ResultMeasureValue", "ActivityStartDate"]]
        station_data=station_data.set_index("ActivityStartDate").rolling("365d").mean(center=True)
        station_data.plot(ax=ax, label=st)
    
```

```python

```

```python

```

# Trends by the Season

```python
despiked_agg
```

```python
seasons = dict(zip(np.arange(1,13), ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]))
```

```python
#Simple Approach
seasonal_ph=despiked_agg.loc[despiked_agg["OrganizationIdentifier"]=="CT_DEP01_WQX"].copy(deep=True)
seasonal_ph["Month"]=seasonal_ph["Date"].dt.month
grouped=seasonal_ph.groupby(["area", "Month"])["ResultMeasureValue"].mean().unstack(level=[1])
grouped.index.name="Basin"
grouped.index=grouped.index.astype(int)
grouped.columns=seasons.values()
grouped
```

```python
print(seasons)
for season in np.arange(1,13):
    for basin in np.arange(1, 6):
        month_data=despiked_agg.loc[(despiked_agg["OrganizationIdentifier"]=="CT_DEP01_WQX") & (despiked_agg["ActivityStartDate"].dt.month==season) & (despiked_agg["area"]==basin)].copy(deep=True)
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
                plt.plot(month_data.loc[month_data["MonitoringLocationIdentifier"]==st, "year"], month_data.loc[month_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"], label=st_names[st])
            plt.legend(title="Station", loc='upper right')
            plt.savefig("Charts-09-07/" + org + "/" + seasons[season] + " " + org +".png", bbox_inches='tight')            
            plt.show()

```

```python

```

# Importing DO Data


## Processing

```python
do=pd.read_csv("../Data Visualization and Analytics Challenge/Data/CT_DEEP_Post_2010_1.12.2023.csv")
do.head()
```

```python
#Dropping na values and sort by date and time
do.drop("Unnamed: 0", axis=1, inplace=True)
print(len(do))
do.dropna(subset="Dissolved Oxygen", inplace=True)
print(len(do))
do
```

```python
do.sort_values(["station name", "Date", "depth"], inplace=True)
do.head()
```

```python
#Converting depths to numeric
do["depth"]=pd.to_numeric(do["depth"], errors="coerce")
print(len(do))
do.dropna(subset="depth", axis=0, inplace=True)
print(len(do))
```

```python
#Making format uniform between WQX ph and do
do["MonitoringLocationIdentifier"]="CT_DEP01_WQX-" + do["station name"].astype(str)

```

```python
#Dates
do["Date"]=pd.to_datetime(do["Date"])
do["Year"]=do["Date"].dt.year
```

```python
#Getting basins
do=do.merge(stations[["MonitoringLocationIdentifier", "loc", "area"]], how="left", on="MonitoringLocationIdentifier")
```

```python
#Investigating missing basins (no idea what I2 missing means)
pd.unique(do.loc[do["area"].isna(), "MonitoringLocationIdentifier"])
```

## Breaking down by depth section

```python
#Repeating the procedure from temp time series
do_dict={}
for station in pd.unique(year_round):
    working=do.loc[do["MonitoringLocationIdentifier"]==station].copy(deep=True)

    #saving
    do_dict[station]=working
    
```

```python
#Length of data from all year-round stations
sum([len(df) for df in do_dict.values()])
```

```python
#Seperating by segment of the depth profile
do_depth_dict={}
for i in np.arange(int(do["depth"].max())+1):
    do_depth_dict[i]={}
    subdict=do_depth_dict[i]
    for index, df in do_dict.items():
        working=df.copy(deep=True)
        working=working.loc[(working["depth"]>=i) & (working["depth"]<(i+1))]
        subdict[index]=working
```

```python
depths=do_depth_dict.keys()
depths
```

```python
#Graphing param
drange=3
```

```python
#Graphing
for basin in np.arange(1, 6):
    working=do.loc[(do["area"]==basin) & (do["depth"]>=drange) & (do["depth"]<(drange+1))]
    fig, ax = plt.subplots()
    ax.set_title("Basin " + str(basin) + " CTDEEP Stations- Year Round at depth " +"[" + str(drange) + ", " +str(drange + 1)+ ") m")
    for st in year_round:
        org_data=working.set_index("Date")
        if len(org_data.loc[org_data["MonitoringLocationIdentifier"]==st])>0:
            org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "Dissolved Oxygen"].plot(ax=ax, label=st)
    ax.legend()
    
```

```python
#Dropping notable outlier
do=do.loc[~((do["cruise name"]=="WQMAR20")& (do["station name"]=="D3"))]
```

```python
#Graphing
for basin in np.arange(1, 6):
    working=do.loc[(do["area"]==basin) & (do["depth"]>=drange) & (do["depth"]<(drange+1))]
    fig, ax = plt.subplots()
    ax.set_title("Basin " + str(basin) + " CTDEEP Stations- Year Round at depth " +"[" + str(drange) + ", " +str(drange + 1)+ ") m")
    for st in year_round:
        org_data=working.set_index("Date")
        if len(org_data.loc[org_data["MonitoringLocationIdentifier"]==st])>0:
            org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "Dissolved Oxygen"].plot(ax=ax, label=st)
    ax.legend()
    
```

```python
#d3
fig, ax = plt.subplots()
ax.set_title("CTDEEP Stations- D3 (do)")
working=do.set_index("Date")
working.loc[working["MonitoringLocationIdentifier"]=="CT_DEP01_WQX-D3", "Dissolved Oxygen"].plot(ax=ax)
#print(working.loc[working["salinity"]>10000, "salinity"])
```

## Quick Check on Profilers

```python
#Info on profilers
print(pd.unique(do["Profiler"]))
combos=do[["cruise name", "station name","Profiler"]].drop_duplicates()
combos.loc[combos.duplicated(subset=["cruise name", "station name"], keep=False)]
```

```python
#Cruises with multiple profilers
pd.unique(do.loc[do.duplicated(subset=["station name", "cruise name", "depth"]), "cruise name"])
```

```python
#Comparing data from diff profilers
do1=do.loc[(do["Year"]==2017) & (do["Profiler"]=="CTD #2")]
print(len(do1))
do2=do.loc[(do["Year"]==2017) & (do["Profiler"]=="EXO 1314")]
print(len(do2))
```

```python
#graphing param
drange=2
```

```python
#Graphing one profile vs another
for st in year_round:
    working1=do1.loc[(do1["MonitoringLocationIdentifier"]==st) & (do1["depth"]>=drange) & (do1["depth"]<(drange+1))].copy(deep=True)
    working2=do2.loc[(do2["MonitoringLocationIdentifier"]==st) & (do2["depth"]>=drange) & (do2["depth"]<(drange+1))].copy(deep=True)
    working1.sort_values(["station name", "Date", "depth"], inplace=True)
    working2.sort_values(["station name", "Date", "depth"], inplace=True)
    
    fig, ax = plt.subplots()
    ax.set_title(st)
    plt.plot(working1["Date"], working1["Dissolved Oxygen"], label="CTD #2")
    plt.plot(working2["Date"], working2["Dissolved Oxygen"], label="EXO 1314")
    plt.xticks(rotation=90)
    plt.legend(loc="upper right")
    print(working1["Dissolved Oxygen"].mean())
    print(working2["Dissolved Oxygen"].mean())
    plt.show()
```

```python
do1.groupby(["Year", "station name"])["Dissolved Oxygen"].mean().unstack(level=1)
```

```python
do2.groupby(["Year", "station name"])["Dissolved Oxygen"].mean().unstack(level=1)
```

```python
do.groupby(["Year", "station name"])["Dissolved Oxygen"].mean().unstack(level=-1)
```

# Seasonal Decompose


## Overall

```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% ! important; }<style>"))
```

```python
monthly_do=do.set_index("Date").resample('M').mean().ffill()
monthly_do.loc[monthly_do["Dissolved Oxygen"].isna()]
```

```python
from statsmodels.tsa.seasonal import STL
res_do = STL(monthly_do["Dissolved Oxygen"]).fit()
res_do.plot()
plt.show()

```

```python
monthly_ph=despiked_agg.set_index("Date")["ResultMeasureValue"].resample('M').mean().ffill()
monthly_ph.loc[monthly_ph.isna()]
```

```python
from statsmodels.tsa.seasonal import STL
res_ph = STL(monthly_ph).fit()
res_ph.plot()
plt.show()

```

```python
from datetime import datetime
```

```python
#Setting size of figures
plt.rcParams["figure.figsize"]=(21, 5)
```

```python
seasonal_do=res_do.seasonal/max(res_do.seasonal)
seasonal_ph=res_ph.seasonal/max(res_ph.seasonal)
seasonal_do.plot(label="Dissolved Oxygen")
seasonal_ph.plot(label="pH")
plt.legend(loc="upper right", fontsize=20)
ax=plt.gca()
late_summer=[datetime.strptime('Aug 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
new_year=[datetime.strptime('Jan 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
ax.set_xticks(late_summer, minor=True)
ax.set_xticks(new_year, major=True)
ax.xaxis.grid(True, linewidth=2, which='major')
ax.xaxis.grid(True, linewidth=2, which='minor', color="r")
ax.set_ylabel("Standardized Units", fontsize="xx-large")
plt.savefig("../OneDrive - Environmental Protection Agency (EPA)/Downloads/WQMWG Presentation/CTDEEP_do_ph.png", bbox_inches='tight')
plt.show()
```

```python
help(dt.datetime)
```

## By Depth Profile

```python
do.rename(columns={"depth": "Depth"}, inplace=True)
```

```python
for depth in depths:
    #do
    working=do.loc[(do["Depth"]>=drange) & (do["Depth"]<(drange+1))]
    monthly_do=working.set_index("Date").resample('M').mean().ffill()
    
    #ph
    working=despiked_agg.loc[(despiked_agg["Depth"]>=drange) & (despiked_agg["Depth"]<(drange+1))]
    monthly_ph=working.set_index("Date")["ResultMeasureValue"].resample('M').mean().ffill()
    
    #models
    res_do = STL(monthly_do["Dissolved Oxygen"]).fit()
    res_ph = STL(monthly_ph).fit()
    
    #graphing
    seasonal_do=res_do.seasonal/max(res_do.seasonal)
    seasonal_ph=res_ph.seasonal/max(res_ph.seasonal)
    seasonal_do.plot(label="Dissolved Oxygen")
    seasonal_ph.plot(label="pH")
    plt.legend(loc="upper right", fontsize=20)
    ax=plt.gca()
    late_summer=[datetime.strptime('Aug 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
    new_year=[datetime.strptime('Jan 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
    ax.set_xticks(late_summer, minor=True)
    ax.set_xticks(new_year, major=True)
    ax.xaxis.grid(True, linewidth=2, which='major')
    ax.xaxis.grid(True, linewidth=2, which='minor', color="r")
    plt.title("Depth of " + str(depth) + " to " + str(depth + 1) + "m", fontsize="xx-large")
    plt.show()
```

## By Basin

```python
for basin in np.arange(1, 6):
    for st in year_round:    
        #do
        working_do=do.loc[(do["area"]==basin) & (do["MonitoringLocationIdentifier"]==st)]

        #ph
        working_ph=despiked_agg.loc[(despiked_agg["area"]==basin) & (despiked_agg["MonitoringLocationIdentifier"]==st)]
        
        if (len(working_do)>0) & (len(working_ph)>0):
            #Monthly sampling
            monthly_do=working_do.set_index("Date").resample('M').mean().ffill()
            monthly_ph=working_ph.set_index("Date")["ResultMeasureValue"].resample('M').mean().ffill()
            
            #Models
            res_do = STL(monthly_do["Dissolved Oxygen"]).fit()
            res_ph = STL(monthly_ph).fit()

            #graphing
            seasonal_do=res_do.seasonal/abs(min(res_do.seasonal))
            seasonal_ph=res_ph.seasonal/abs(min(res_ph.seasonal))
            seasonal_do.plot(label="Dissolved Oxygen")
            seasonal_ph.plot(label="pH")
            plt.legend(loc="upper right", fontsize=20)
            ax=plt.gca()
            late_summer=[datetime.strptime('Aug 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
            new_year=[datetime.strptime('Jan 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
            ax.set_xticks(late_summer, minor=True)
            ax.set_xticks(new_year, major=True)
            ax.xaxis.grid(True, linewidth=2, which='minor', color="r")
            ax.xaxis.grid(True, linewidth=2, which='major')
            ax.set_title("Basin " + str(basin) + ": " + st_names[st], fontsize="xx-large")
            plt.show()
```

## By Station

```python
for basin in np.arange(1, 6):
    fig, ax = plt.subplots()
    ax.set_title("Basin " + str(basin) + " CTDEEP Stations- Seasonal Trends")
    for st in year_round:
        org_data=working.set_index("ActivityStartDate")
        org_data.loc[org_data["MonitoringLocationIdentifier"]==st, "ResultMeasureValue"].plot(ax=ax)
    
        #do
        working_do=do.loc[(do["area"]==basin) & (do["MonitoringLocationIdentifier"]==st)]
        monthly_do=working_do.set_index("Date").resample('M').mean().ffill()

        #ph
        working_ph=despiked_agg.loc[(despiked_agg["area"]==basin) & (despiked_agg["MonitoringLocationIdentifier"]==st)]
        monthly_ph=working_ph.set_index("Date")["ResultMeasureValue"].resample('M').mean().ffill()
        if len(working_do)>0:
        #models
        res_do = STL(monthly_do["Dissolved Oxygen"]).fit()
        res_ph = STL(monthly_ph).fit()

        #graphing
        seasonal_do=res_do.seasonal/max(res_do.seasonal)
        seasonal_ph=res_ph.seasonal/max(res_ph.seasonal)
        seasonal_do.plot(label="Dissolved Oxygen")
        seasonal_ph.plot(label="pH")
        plt.legend(loc="upper right", fontsize=20)
        ax=plt.gca()
        late_summer=[datetime.strptime('Aug 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
        new_year=[datetime.strptime('Jan 1 ' + str(year), '%b %d %Y') for year in list(pd.unique(do["Year"]))]
        ax.set_xticks(late_summer, minor=True)
        ax.set_xticks(new_year, major=True)
        ax.xaxis.grid(True, linewidth=2, which='major')
        ax.xaxis.grid(True, linewidth=2, which='minor', color="r")
        plt.title("Depth of " + str(depth) + " to " + str(depth + 1) + "m")
        plt.show()
```

```python

```

# Rolling Averages (Old Version)

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
