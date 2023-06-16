```python
import os
import requests
import numpy as np
import requests, zipfile, io
import csv
```

```python
strt_yr=2010
end_yr=2022
query_name="query_2"
query_start="https://www.waterqualitydata.us/data/Result/search?countrycode=US&within=70&lat=41.12&long=-72.81&siteType=Estuary&sampleMedia=Water&sampleMedia=water&characteristicName=PH&characteristicName=pH&"
query_end="&mimeType=csv&zip=yes&dataProfile=resultPhysChem&providers=NWIS&providers=STEWARDS&providers=STORET"
```

```python
if strt_yr<=end_yr:
    prd_strts=list()
    prd_ends=list()
    for year in np.arange(strt_yr, end_yr+1):
        prd_strt="01-01-"+ str(year)
        prd_end="01-01-"+ str(year+1)
        
        prd_strts.append(prd_strt)
        prd_ends.append(prd_end)

print(prd_strts)
print(prd_ends)
```

```python
for i in np.arange(len(prd_strts)):
    try:
        r = requests.get(query_start
                      + "startDateLo=" + prd_strts[i] + "&startDateHi=" + prd_ends[i] + 
                     query_end,
                     stream=True)
    except:
        print("no url for period from " + prd_strts[i] + " to " + prd_ends[i])
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("WQP_Data/"+query_name + "/" + prd_strts[i][-4:]+"_"+ prd_ends[i][-4:] )
```

```python
print(prd_strts[i][-4:])
```

```python

```
