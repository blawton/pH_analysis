```python
import numpy as np
import pandas as pd
```

```python
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
```

```python
df=pd.read_csv("WQP_Data/resultphyschem.csv")
df["ActivityDepthHeightMeasure/MeasureValue"].head(20)
```

```python
df.head()
```

```python
print(df["ActivityStartDate"].max())
print(df["ActivityStartDate"].min())
```

```python
print(len(df))
```

```python
ph_df=df.loc[df["CharacteristicName"]=="pH"]
print(len(ph_df))
```

```python

```
