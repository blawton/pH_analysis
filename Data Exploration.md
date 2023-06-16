```python
import numpy as np
import pandas as pd
```

```python
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
```

# Reading in Data From WQP and excel

```python
strt_yr=2010
end_yr=2022
```

```python
agg_data=pd.DataFrame()

for i in np.arange(strt_yr, end_yr):
    print(len(agg_data.columns))
    print(len(agg_data))
    current_year=pd.read_csv("WQP_Data/query_2/"+ str(i)+ "_" + str(i+1) +"/resultphyschem.csv")
    agg_data=pd.concat([agg_data, current_year])
agg_data.reset_index(inplace=True, drop=True)
agg_data.head()
```

```python
agg_data["year"]=agg_data["ActivityStartDate"].str[0:4]
agg_data.head()
```

```python
#Dropping nas in OrganizationIdentifier
print(len(agg_data))
agg_data.dropna(subset="OrganizationIdentifier", inplace=True)
print(len(agg_data))
```

# Adding in CTDEEP Data from Excel

```python
# Reading in excel data
excel_ct=pd.read_csv("CTDEEP_data_from_excel.csv", index_col=0)
excel_ct.head() 
```

```python
# Time distrubution of data from WQP
agg_data.groupby("year")["ResultMeasureValue"].count()
```

```python
# Time distrubution of data from Excel
excel_ct.groupby("year")["ResultMeasureValue"].count()
```

```python
#Checking for dupes
print(len(agg_data.loc[agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX"]))
```

```python
#Depth range
agg_data.loc[agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX", "ActivityDepthHeightMeasure/MeasureValue"].max()-agg_data.loc[agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX", "ActivityDepthHeightMeasure/MeasureValue"].min()
```

```python
discard=list(set(excel_ct.columns) - set(agg_data.columns))
print(discard)
overlap = [x for x in excel_ct.columns if x in agg_data.columns]
overlap = overlap + ["cruise name"]
print(overlap)
```

```python
agg_data=pd.concat([agg_data, excel_ct[overlap]], ignore_index=True)
agg_data.head()
```

# Getting CTDEEP Lon/Lat and Station Identifiers

```python
#Fixing station names based on info from Katie O'Brien Clayton at CTDEEP
agg_data.loc[agg_data["MonitoringLocationName"]=="29B", "MonitoringLocationName"]="29"
agg_data.loc[agg_data["MonitoringLocationName"]=="26B", "MonitoringLocationName"]="26"
agg_data.loc[agg_data["MonitoringLocationName"]=="30B", "MonitoringLocationName"]="30"
agg_data.loc[agg_data["MonitoringLocationName"]=="32", "MonitoringLocationIdentifier"]="CT_DEP01_WQX-32"

```

```python
#Getting longitutde and lattitude of CTDEEP data
station_map= agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (~agg_data["MonitoringLocationIdentifier"].isna()), ["MonitoringLocationIdentifier", "MonitoringLocationName", "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"]].drop_duplicates()
print(station_map)

#Ensuring identifier is unique
print(len(station_map))
print(len(station_map.drop_duplicates(subset="MonitoringLocationIdentifier")))

#Getting new stations without a match in WQP Data
new_stations = agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data["MonitoringLocationIdentifier"].isna()), ["MonitoringLocationIdentifier", "MonitoringLocationName", "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"]].drop_duplicates()
print(list(set(new_stations["MonitoringLocationName"])-set(station_map["MonitoringLocationName"])))
print(len(new_stations))
```

```python
#Performing the mapping
for index, station in station_map.iterrows():
    agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["MonitoringLocationName"]==station["MonitoringLocationName"]), "MonitoringLocationIdentifier"]=station["MonitoringLocationIdentifier"]
    agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["MonitoringLocationName"]==station["MonitoringLocationName"]), "ActivityLocation/LatitudeMeasure"]=station["ActivityLocation/LatitudeMeasure"]
    agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["MonitoringLocationName"]==station["MonitoringLocationName"]), "ActivityLocation/LongitudeMeasure"]=station["ActivityLocation/LongitudeMeasure"]
    

    
```

```python
#Testing if the mapping worked (these stations should be the unmatched ones above)
badnames=agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data["MonitoringLocationIdentifier"].isna()), "MonitoringLocationName"].drop_duplicates()
print(badnames)
```

```python
#Getting rid of extraneous stations (as per info from CTDEEP)
agg_data.drop(agg_data.loc[agg_data["MonitoringLocationName"].isin(badnames)].index, axis=0, inplace=True)
#Test
agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data["MonitoringLocationIdentifier"].isna()), "MonitoringLocationName"]

```

# Getting Rid of Duplicates

```python
#Ensuring "year" and other columns are numeric to properly group and identify duplicates
agg_data['year']=pd.to_numeric(agg_data['year'], errors='coerce')
agg_data['ActivityDepthHeightMeasure/MeasureValue']=pd.to_numeric(agg_data['ActivityDepthHeightMeasure/MeasureValue'], errors='coerce')

```

```python
#Identifying dupes and checking to ensure times are comparable
dupes = agg_data.loc[agg_data.duplicated(subset=["ActivityStartDate", "MonitoringLocationIdentifier", "ActivityDepthHeightMeasure/MeasureValue"], keep='first')]
print(len(dupes))
dupes2 = agg_data.loc[agg_data.duplicated(subset=["ActivityStartDate", "MonitoringLocationIdentifier", "ActivityDepthHeightMeasure/MeasureValue", "ActivityStartTime/Time"], keep='first')]
print(len(dupes2))
dupes3 = agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data.duplicated(subset=["ActivityStartDate", "MonitoringLocationIdentifier", "ActivityDepthHeightMeasure/MeasureValue", "ActivityStartTime/Time"], keep='first'))]
print(len(dupes3))

print(pd.unique(dupes["OrganizationIdentifier"]))
print(pd.unique(dupes2["OrganizationIdentifier"]))
print(pd.unique(dupes3["OrganizationIdentifier"]))
```

```python
# Breakdown of Duplicates by Year (for comparison with CTDEEP Breakdown above)
print(dupes.groupby('year')["ResultMeasureValue"].count())
print(dupes2.groupby('year')["ResultMeasureValue"].count())
print(dupes3.groupby('year')["ResultMeasureValue"].count())
```

# Further Duplicate Testing

```python
duped = agg_data.loc[agg_data.duplicated(subset=["ActivityStartDate", "MonitoringLocationIdentifier", "ActivityDepthHeightMeasure/MeasureValue"], keep=False)]
print(len(duped))
duped2 = agg_data.loc[agg_data.duplicated(subset=["ActivityStartDate", "MonitoringLocationIdentifier", "ActivityDepthHeightMeasure/MeasureValue", "ActivityStartTime/Time"], keep=False)]
print(len(duped2))
duped3 = agg_data.loc[(agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX") & (agg_data.duplicated(subset=["ActivityStartDate", "MonitoringLocationIdentifier", "ActivityDepthHeightMeasure/MeasureValue", "ActivityStartTime/Time"], keep=False))]
print(len(duped3))
```

```python
#Getting info to determine which set of dupes to drop (_, 2, or 3)
fringedupes=duped3.copy(deep=True)
fringedupes.sort_values(by=["MonitoringLocationIdentifier", "ActivityDepthHeightMeasure/MeasureValue", "ActivityStartDate", "ActivityStartTime/Time"], inplace=True)
fringedupes.tail(50)
```

```python
og_len=len(agg_data)
```

```python
agg_data.drop(dupes3.index, inplace=True)
print(len(agg_data))
print(og_len - len(agg_data))
```

```python
print(og_len)
```

```python
agg_data.count()/len(agg_data)*100
```

# Adding IEC Data from excel

```python
#Reading in IEC data from excel
excel_df=pd.read_csv("IEC_data_from_excel.csv", index_col=0)
excel_df.head() 
```

```python
discard=list(set(excel_df.columns) - set(agg_data.columns))
print(discard)
overlap = [x for x in excel_df.columns if x in agg_data.columns]
print(overlap)
```

```python
#Adding in IEC data from excel
agg_data=pd.concat([agg_data, excel_df[overlap]], ignore_index=True)
agg_data.head()
```

```python
agg_data.count()/len(agg_data)*100
```

```python
#Getting more specific names for Western Long Island Sound Locations (from more recent IEC data)
#print(agg_data.loc[agg_data["MonitoringLocationName"]=="Western Long Island Sound", ["MonitoringLocationIdentifier", "ActivityLocation/LatitudeMeasure", "ActivityLocation/LatitudeMeasure"]])
western_stations=agg_data.loc[agg_data["MonitoringLocationName"]=="Western Long Island Sound", ["MonitoringLocationIdentifier", "MonitoringLocationName"]]
finer_names=agg_data.loc[agg_data["MonitoringLocationIdentifier"].isin(western_stations["MonitoringLocationIdentifier"]), ["MonitoringLocationIdentifier", "MonitoringLocationName"]].drop_duplicates(subset="MonitoringLocationName")
finer_names=finer_names.loc[finer_names["MonitoringLocationName"]!="Western Long Island Sound"]
finer_names = pd.concat([finer_names, western_stations.loc[~western_stations["MonitoringLocationIdentifier"].isin(finer_names["MonitoringLocationIdentifier"])].drop_duplicates(subset="MonitoringLocationIdentifier")], axis=0)

#Fixing index for next step
finer_names.columns=["MonitoringLocationIdentifier", "FinerName"]
print(finer_names)

#Uniformly mapping data to finer names
agg_data=agg_data.merge(finer_names, how='left', on="MonitoringLocationIdentifier")
agg_data.loc[~agg_data["FinerName"].isna(), "MonitoringLocationName"]= agg_data["FinerName"]
```

```python
#Test of refined mapping
agg_data.loc[~agg_data["FinerName"].isna()].head(10)
```

```python
#Dropping unnec. columns
agg_data.drop("FinerName", axis=1, inplace=True)
agg_data.head()
```

# USGS Coords

```python
#Finding number of unique monitoring locations
usgs_stations=agg_data.loc[(agg_data["OrganizationIdentifier"]=="USGS-CT") | (agg_data["OrganizationIdentifier"]=="USGS-NY"), "MonitoringLocationIdentifier"].drop_duplicates()
print(usgs_stations)
```

```python
import requests
from io import StringIO
from IPython import display
```

```python
#rdb reader from hydrofunctions for usgs file types
def read_rdb(text):
    """Read strings that are in rdb format.

    Args:
        text (str):
            A long string containing the contents of a rdb file. A common way
            to obtain these would be from the .text property of a requests
            response, as in the example usage below.

    Returns:
        header (multi-line string):
            Every commented line at the top of the rdb file is marked with a
            '#' symbol. Each of these lines is stored in this output.
        outputDF (pandas.DataFrame):
            A dataframe containing the information in the rdb file. `site_no`
            and `parameter_cd` are interpreted as a string, but every other number
            is interpreted as a float or int; missing values as an np.nan;
            strings for everything else.
        columns (list of strings):
            The column names, taken from the rdb header row.
        dtypes (list of strings):
            The second header row from the rdb file. These mostly tell the
            column width, and typically record everything as string data ('s')
            type. The exception to this are dates, which are listed with a 'd'.
    """
    try:
        headerlines = []
        datalines = []
        count = 0
        for line in text.splitlines():
            if line[0] == "#":
                headerlines.append(line)
            elif count == 0:
                columns = line.split()
                count += 1
            elif count == 1:
                dtypes = line.split()
                count += 1
            else:
                datalines.append(line)
        data = "\n".join(datalines)
        header = "\n".join(headerlines)

        outputDF = pd.read_csv(
            StringIO(data),
            sep="\t",
            comment="#",
            header=None,
            names=columns,
            dtype={"site_no": str, "parameter_cd": str},
            # When converted like this, poorly-formed dates will be saved as strings
            parse_dates=True,
            # If dates are converted like this, then poorly formed dates will stop the process
            # converters={"peak_dt": pd.to_datetime},
        )
        # Another approach would be to convert date columns later, and catch errors
        # try:
        #   outputDF.peak_dt = pd.to_datetime(outputDF.peak_dt)
        # except ValueError as err:
        #   print(f"Unable to parse date. reason: '{str(err)}'.")

    except:
        print(
            "There appears to be an error processing the file that the USGS "
            "returned. This sometimes occurs if you entered the wrong site "
            "number. We were expecting an RDB file, but we received the "
            f"following instead:\n{text}"
        )
        raise
    # outputDF.outputDF.filter(like='_cd').columns
    # TODO: code columns ('*._cd') should be interpreted as strings.
    # TODO: date columns ('*_dt') should be converted to dates.

    return header, outputDF, columns, dtypes
```

```python
#Testing function and getting templates for df
testdf = read_rdb(requests.get('https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=01196530&siteStatus=all').text)[1]
print(testdf)

```

```python
usgs_df = pd.DataFrame()

for station in usgs_stations.values:
    url = 'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=' + str(station)[5:] + '&siteStatus=all'
    text= requests.get(url).text
    df = read_rdb(text)[1]
    usgs_df=pd.concat([usgs_df, df])

usgs_df['site_no']=usgs_df['site_no'].map(lambda x: 'USGS-'+ str(x))
usgs_df.rename(columns={'site_no':'MonitoringLocationIdentifier'}, inplace=True)
usgs_df.head()
```

```python
#putting coords and names into agg_data
merged = agg_data.merge(usgs_df, how='left', on='MonitoringLocationIdentifier')

#Putting CT data in
agg_data.loc[agg_data["OrganizationIdentifier"]=="USGS-CT", "ActivityLocation/LatitudeMeasure"]=merged['dec_lat_va']
agg_data.loc[agg_data["OrganizationIdentifier"]=="USGS-CT", "ActivityLocation/LongitudeMeasure"]=merged['dec_long_va']
agg_data.loc[agg_data["OrganizationIdentifier"]=="USGS-CT", "MonitoringLocationName"]=merged['station_nm']

#Putting NY data in
agg_data.loc[agg_data["OrganizationIdentifier"]=="USGS-NY", "ActivityLocation/LatitudeMeasure"]=merged['dec_lat_va']
agg_data.loc[agg_data["OrganizationIdentifier"]=="USGS-NY", "ActivityLocation/LongitudeMeasure"]=merged['dec_long_va']
agg_data.loc[agg_data["OrganizationIdentifier"]=="USGS-NY", "MonitoringLocationName"]=merged['station_nm']

agg_data.loc[agg_data["OrganizationIdentifier"]=="USGS-NY"].head()
```

```python
#fixing year inconsistencies
agg_data.loc[pd.to_numeric(agg_data['year'], errors='coerce').isnull()]
agg_data.loc[agg_data["ActivityIdentifier"]=="31ISC2RS_WQX-9-409:20210722-53800-pH D:0.5", 'year']=2021
agg_data.loc[pd.to_numeric(agg_data['year'], errors='coerce').isnull()]
```

```python
agg_data.loc[122450]
```

```python
#fixing number format inconsistencies
agg_data['year']=agg_data['year'].astype('float')
agg_data['year']=agg_data['year'].map('{:g}'.format)
agg_data['year']=agg_data['year'].astype('int')
```

```python
len(agg_data)
```

# Getting the Right Station Identifier for all stations

```python
#Extent of problem
print(agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()), "OrganizationIdentifier"].drop_duplicates())
agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX")]
```

```python
agg_data.loc[(~agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX")]
```

```python
#CTDEP Data First
print(agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX"), "MonitoringLocationName"].map(lambda x: "CT_DEP01_WQX-" + str(x)))
agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX"), "MonitoringLocationIdentifier"] = agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX"), "MonitoringLocationName"].map(lambda x: "CT_DEP01_WQX-" + str(x))
agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX")]
```

# IEC Stations (if neccesary)

```python
#Making sure all IEC Stations have a station identifier (if identified as problem above)
missing_locs = agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX"), "ActivityIdentifier"].copy(deep=True)
missing_locs=missing_locs.map(lambda x: x.lstrip("31ISC2RS_WQX-")).str.split(":", expand=True)
print(missing_locs)
missing_locs=missing_locs.map(lambda x: "31ISC2RS_WQX-"+x)
print(missing_locs)
#agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX"), "MonitoringLocationIdentifier"]=missing_locs
#agg_data.loc[(agg_data["MonitoringLocationIdentifier"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX")]

```

# Making sure Station Identifiers Match Up

```python
#Quick check on CTDEP Station Identifiers
agg_data.loc[agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX", "MonitoringLocationIdentifier"].drop_duplicates()
```

```python
#Remapping CTDEEP Stations based on Katie Instructions
NCCAs=["CT_DEP01_WQX-0219", "CT_DEP01_WQX-0422", "CT_DEP01_WQX-5914", "CT_DEP01_WQX-0210", "CT_DEP01_WQX-0313", "CT_DEP01_WQX-0617", "CT_DEP01_WQX-5911"]
agg_data.drop(agg_data.loc[agg_data["MonitoringLocationIdentifier"].isin(NCCAs)].index, inplace=True)
agg_data.replace({"CT_DEP01_WQX-26B":"CT_DEP01_WQX-26", "CT_DEP01_WQX-29B":"CT_DEP01_WQX-29", "CT_DEP01_WQX-30B":"CT_DEP01_WQX-30"}, inplace=True)
agg_data.loc[agg_data["OrganizationIdentifier"]=="CT_DEP01_WQX", "MonitoringLocationIdentifier"].drop_duplicates()

```

```python
#IEC Data Station Identifiers

#Ensuring all stations with no lat/lon are covered by earlier data and all mismatch is from naming discrepancies
nyears = agg_data.loc[(agg_data["ActivityLocation/LatitudeMeasure"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX"), "year"].drop_duplicates()
missing_coord = agg_data.loc[(agg_data["ActivityLocation/LatitudeMeasure"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX"), "MonitoringLocationIdentifier"].drop_duplicates()
with_coord = agg_data.loc[(~agg_data["ActivityLocation/LatitudeMeasure"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX"), "MonitoringLocationIdentifier"].drop_duplicates()

#Matching station names in excel data with existing names
changed_names=list(set(missing_coord.values)-set(with_coord.values))
with_coord=pd.DataFrame(with_coord)
with_coord.columns=["proper_name"]
with_coord["no_dash"] = with_coord["proper_name"].str.replace("-", "")
name_map = pd.DataFrame(changed_names, columns=["name"])
name_map["no_dash"]=name_map["name"].str.replace("-", "")
name_map=name_map.merge(with_coord, how="left", on="no_dash")

print(with_coord)
print(missing_coord)
print(changed_names)
print(name_map.head(10))


```

```python
#Switching out new names for the old

print(len(agg_data.loc[agg_data["MonitoringLocationIdentifier"].isin(changed_names)]))

for index, row in name_map.iterrows():
    agg_data.loc[agg_data["MonitoringLocationIdentifier"]==row["name"], "MonitoringLocationIdentifier"]= row["proper_name"]

agg_data.loc[agg_data["MonitoringLocationIdentifier"].isin(changed_names)]
```

```python
#Making sure all stations have a coordinate somewhere (not neccesarily for all entries)
missing_coord = agg_data.loc[(agg_data["ActivityLocation/LatitudeMeasure"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX"), "MonitoringLocationIdentifier"].drop_duplicates()
with_coord = agg_data.loc[(~agg_data["ActivityLocation/LatitudeMeasure"].isna()) & (agg_data["OrganizationIdentifier"]=="31ISC2RS_WQX"), "MonitoringLocationIdentifier"].drop_duplicates()
list(set(missing_coord.values)-set(with_coord.values))
```

```python
#Exporting Data to csv 
agg_data.to_csv('cleaned_aggregated_pH_data.csv')
```

```python
len(agg_data)
```

# Honing in on useful data sources

```python
#IEC data
agg_data.loc[agg_data["MonitoringLocationIdentifier"]=="31ISC2RS_WQX-9-413"]
```

```python
print(agg_data.groupby(["year"])["ResultMeasureValue"].count())
print(agg_data.groupby(["year"])["ActivityMediaName"].count())
```

```python
counted=agg_data.groupby(["year", "OrganizationIdentifier"])["ResultMeasureValue"].count()
print(len(counted))
counted.unstack(0).head(15)
```

```python
key=agg_data[["OrganizationIdentifier", "OrganizationFormalName"]].drop_duplicates()
print(key)
```

# Data Availability

```python
#Dropping nas from measurement value field
print(len(agg_data))
agg_data=agg_data.dropna(subset=["ResultMeasureValue"])
print(len(agg_data))
```

```python
#Simple counts of each measurement location from organizations with enough data history to measure
usable_orgs= ["11NPSWRD_WQX", "31ISC2RS_WQX", "NJHDG", "SCDHSECOLOGY", "USGS-CT", "CT_DEP01_WQX"]
for org in usable_orgs:
    print(org)
    print(agg_data.loc[agg_data["OrganizationIdentifier"]==org].groupby("MonitoringLocationName")["ResultMeasureValue"].count())

```

```python
#More refined counts showing locations and years
for org in usable_orgs:
    org_data=agg_data.loc[agg_data["OrganizationIdentifier"]==org, ["year", "MonitoringLocationName"]]
    pivot=org_data.pivot_table(index="MonitoringLocationName", columns="year", aggfunc=len)
    print(org)
    print(pivot)
```

```python
#Variances of lattitude and longitude within each location identifier
for org in usable_orgs:
    org_data=agg_data.loc[agg_data["OrganizationIdentifier"]==org, ["MonitoringLocationName", "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"]]
    pivot=org_data.pivot_table(index="MonitoringLocationName", aggfunc=np.var)
    print(org)
    print(pivot)
```

# Depth

```python
#Getting a little more data on depth
print(len(agg_data))
print(len(agg_data.dropna(subset="ActivityDepthHeightMeasure/MeasureValue")))
```

```python
#Getting variances in depth
for org in usable_orgs:
    agg_data_w_depth=agg_data.dropna(subset="ActivityDepthHeightMeasure/MeasureValue")
    org_data=agg_data_w_depth.loc[agg_data_w_depth["OrganizationIdentifier"]==org, ["MonitoringLocationName", "ActivityDepthHeightMeasure/MeasureValue"]]
    pivot=org_data.pivot_table(index="MonitoringLocationName", aggfunc=np.var)
    print(org)
    print(pivot)
```

```python
help(pd.DataFrame.pivot_table)
```

```python
df=pd.read_csv("WQP_Data/query_2/2017_2018/resultphyschem.csv")
df.head()
```

```python
print(df["ActivityStartDate"].max())
print(df["ActivityStartDate"].min())
```

```python
for
```

```python
ph_df=df.loc[df["CharacteristicName"]=="pH"]
print(len(ph_df))
```

```python
help(pd.DataFrame.dropna)
```

```python

```
