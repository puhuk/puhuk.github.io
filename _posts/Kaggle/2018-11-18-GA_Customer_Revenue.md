---
title: "Kaggle - GA Customer Revenue prediction - EDA"
date: 2018-11-18 08:26:28 -0400
categories: Kaggle practice
use_math: true
---

There is an ongoing competition in Kaggle. I want to share the exploration of the data as learned from other competitor's kernel. Of course I will not include the detailed feature engineering and modeling here and now.

### Overview
#### What to predict
Mission of this competition is to predict natural log of **the sum of all transactions per user**
#### Data
- fullVisitorId- User ID
- channelGrouping - Channel
- date - date
- device - device
- geoNetwork - User's geography info
- socialEngagementType - Either "Socially Engaged" or "Not Socially Engaged"
- totals - Aggregate values across the session
- trafficSource - Traffic Source from which the session originated
- visitId - An identifier for this session
- visitNumber - The session number for this user
- visitStartTime - The timestamp
- hits - A record of all page visits
- customDimensions - User-level or session-level custom dimensions that are set for a session
- totals - High-level aggregate data.

#### Evaluation
Using RMSE to measure the difference between the predictions and the fare.
$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2}
$$

So here, the lower RMSE means better prediction.


#### EDA

I want to share that I learned and practiced after study [SRK](https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue) and Andrew [Lukyanenko](https://www.kaggle.com/artgor/fork-of-eda-on-basic-data-and-lgb-in-progress) 's kernel.
Thanks for their open kernel for study and let me start the practice.

Import python library to be used.
```python
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import json
import bq_helper
from pandas.io.json import json_normalize
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
```

Let's see the train data. There are JSON format in several columns like device, hits, etc.

```python
train1=pd.read_csv("../input/train_v2.csv",nrows=200)
train1.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right; {font-size: 5pt;}">
      <th></th>
      <th>channelGrouping</th>
      <th>customDimensions</th>
      <th>date</th>
      <th>device</th>
      <th>fullVisitorId</th>
      <th>geoNetwork</th>
      <th>hits</th>
      <th>socialEngagementType</th>
      <th>totals</th>
      <th>trafficSource</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>20171016</td>
      <td>{"browser": "Firefox", "browserVersion": "not ...</td>
      <td>3162355547410993243</td>
      <td>{"continent": "Europe", "subContinent": "Weste...</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '17',...</td>
      <td>Not Socially Engaged</td>
      <td>{"visits": "1", "hits": "1", "pageviews": "1",...</td>
      <td>{"campaign": "(not set)", "source": "google", ...</td>
      <td>1508198450</td>
      <td>1</td>
      <td>1508198450</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Referral</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>20171016</td>
      <td>{"browser": "Chrome", "browserVersion": "not a...</td>
      <td>8934116514970143966</td>
      <td>{"continent": "Americas", "subContinent": "Nor...</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '10',...</td>
      <td>Not Socially Engaged</td>
      <td>{"visits": "1", "hits": "2", "pageviews": "2",...</td>
      <td>{"referralPath": "/a/google.com/transportation...</td>
      <td>1508176307</td>
      <td>6</td>
      <td>1508176307</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Direct</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>20171016</td>
      <td>{"browser": "Chrome", "browserVersion": "not a...</td>
      <td>7992466427990357681</td>
      <td>{"continent": "Americas", "subContinent": "Nor...</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '17',...</td>
      <td>Not Socially Engaged</td>
      <td>{"visits": "1", "hits": "2", "pageviews": "2",...</td>
      <td>{"campaign": "(not set)", "source": "(direct)"...</td>
      <td>1508201613</td>
      <td>1</td>
      <td>1508201613</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>20171016</td>
      <td>{"browser": "Chrome", "browserVersion": "not a...</td>
      <td>9075655783635761930</td>
      <td>{"continent": "Asia", "subContinent": "Western...</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '9', ...</td>
      <td>Not Socially Engaged</td>
      <td>{"visits": "1", "hits": "2", "pageviews": "2",...</td>
      <td>{"campaign": "(not set)", "source": "google", ...</td>
      <td>1508169851</td>
      <td>1</td>
      <td>1508169851</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'Central America'}]</td>
      <td>20171016</td>
      <td>{"browser": "Chrome", "browserVersion": "not a...</td>
      <td>6960673291025684308</td>
      <td>{"continent": "Americas", "subContinent": "Cen...</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '14',...</td>
      <td>Not Socially Engaged</td>
      <td>{"visits": "1", "hits": "2", "pageviews": "2",...</td>
      <td>{"campaign": "(not set)", "source": "google", ...</td>
      <td>1508190552</td>
      <td>1</td>
      <td>1508190552</td>
    </tr>
  </tbody>
</table>
</div>



[Julian Peller](https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields) opened his kernel to deal with parsing JSON as below.


```python
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
```


```python
%%time
train = load_df("../input/train_v2.csv",nrows=20000)
test = load_df("../input/test_v2.csv")
```

    Loaded train_v2.csv. Shape: (20000, 59)
    Loaded test_v2.csv. Shape: (401589, 59)
    CPU times: user 2min 10s, sys: 11.8 s, total: 2min 22s
    Wall time: 2min 24s
    

After parsing, data looks as below.


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channelGrouping</th>
      <th>customDimensions</th>
      <th>date</th>
      <th>fullVisitorId</th>
      <th>hits</th>
      <th>socialEngagementType</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device.browser</th>
      <th>device.browserSize</th>
      <th>device.browserVersion</th>
      <th>device.deviceCategory</th>
      <th>device.flashVersion</th>
      <th>device.isMobile</th>
      <th>device.language</th>
      <th>device.mobileDeviceBranding</th>
      <th>device.mobileDeviceInfo</th>
      <th>device.mobileDeviceMarketingName</th>
      <th>device.mobileDeviceModel</th>
      <th>device.mobileInputSelector</th>
      <th>device.operatingSystem</th>
      <th>device.operatingSystemVersion</th>
      <th>device.screenColors</th>
      <th>device.screenResolution</th>
      <th>geoNetwork.city</th>
      <th>geoNetwork.cityId</th>
      <th>geoNetwork.continent</th>
      <th>geoNetwork.country</th>
      <th>geoNetwork.latitude</th>
      <th>geoNetwork.longitude</th>
      <th>geoNetwork.metro</th>
      <th>geoNetwork.networkDomain</th>
      <th>geoNetwork.networkLocation</th>
      <th>geoNetwork.region</th>
      <th>geoNetwork.subContinent</th>
      <th>totals.bounces</th>
      <th>totals.hits</th>
      <th>totals.newVisits</th>
      <th>totals.pageviews</th>
      <th>totals.sessionQualityDim</th>
      <th>totals.timeOnSite</th>
      <th>totals.totalTransactionRevenue</th>
      <th>totals.transactionRevenue</th>
      <th>totals.transactions</th>
      <th>totals.visits</th>
      <th>trafficSource.adContent</th>
      <th>trafficSource.adwordsClickInfo.adNetworkType</th>
      <th>trafficSource.adwordsClickInfo.criteriaParameters</th>
      <th>trafficSource.adwordsClickInfo.gclId</th>
      <th>trafficSource.adwordsClickInfo.isVideoAd</th>
      <th>trafficSource.adwordsClickInfo.page</th>
      <th>trafficSource.adwordsClickInfo.slot</th>
      <th>trafficSource.campaign</th>
      <th>trafficSource.isTrueDirect</th>
      <th>trafficSource.keyword</th>
      <th>trafficSource.medium</th>
      <th>trafficSource.referralPath</th>
      <th>trafficSource.source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>20171016</td>
      <td>3162355547410993243</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '17',...</td>
      <td>Not Socially Engaged</td>
      <td>1508198450</td>
      <td>1</td>
      <td>1508198450</td>
      <td>Firefox</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Europe</td>
      <td>Germany</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>(not set)</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>water bottle</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Referral</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>20171016</td>
      <td>8934116514970143966</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '10',...</td>
      <td>Not Socially Engaged</td>
      <td>1508176307</td>
      <td>6</td>
      <td>1508176307</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Chrome OS</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Cupertino</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>United States</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>San Francisco-Oakland-San Jose CA</td>
      <td>(not set)</td>
      <td>not available in demo dataset</td>
      <td>California</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>referral</td>
      <td>/a/google.com/transportation/mtv-services/bike...</td>
      <td>sites.google.com</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Direct</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>20171016</td>
      <td>7992466427990357681</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '17',...</td>
      <td>Not Socially Engaged</td>
      <td>1508201613</td>
      <td>1</td>
      <td>1508201613</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>mobile</td>
      <td>not available in demo dataset</td>
      <td>True</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Android</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>United States</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>windjammercable.net</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>True</td>
      <td>NaN</td>
      <td>(none)</td>
      <td>NaN</td>
      <td>(direct)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>20171016</td>
      <td>9075655783635761930</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '9', ...</td>
      <td>Not Socially Engaged</td>
      <td>1508169851</td>
      <td>1</td>
      <td>1508169851</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Asia</td>
      <td>Turkey</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>unknown.unknown</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Western Asia</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'Central America'}]</td>
      <td>20171016</td>
      <td>6960673291025684308</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '14',...</td>
      <td>Not Socially Engaged</td>
      <td>1508190552</td>
      <td>1</td>
      <td>1508190552</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>Mexico</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>prod-infinitum.com.mx</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Central America</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
  </tbody>
</table>
</div>



Adjust NA of 'trafficSource.adwordsClickInfo.isVideoAd' to True

Adjust NA of 'trafficSource.isTrueDirect' to False

Then set the date format as yyyy-mm-dd


```python
# some data processing
train['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
test['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
train['trafficSource.isTrueDirect'].fillna(False, inplace=True)
test['trafficSource.isTrueDirect'].fillna(False, inplace=True)

train['date'] = pd.to_datetime(train['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
test['date'] = pd.to_datetime(test['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channelGrouping</th>
      <th>customDimensions</th>
      <th>date</th>
      <th>fullVisitorId</th>
      <th>hits</th>
      <th>socialEngagementType</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device.browser</th>
      <th>device.browserSize</th>
      <th>device.browserVersion</th>
      <th>device.deviceCategory</th>
      <th>device.flashVersion</th>
      <th>device.isMobile</th>
      <th>device.language</th>
      <th>device.mobileDeviceBranding</th>
      <th>device.mobileDeviceInfo</th>
      <th>device.mobileDeviceMarketingName</th>
      <th>device.mobileDeviceModel</th>
      <th>device.mobileInputSelector</th>
      <th>device.operatingSystem</th>
      <th>device.operatingSystemVersion</th>
      <th>device.screenColors</th>
      <th>device.screenResolution</th>
      <th>geoNetwork.city</th>
      <th>geoNetwork.cityId</th>
      <th>geoNetwork.continent</th>
      <th>geoNetwork.country</th>
      <th>geoNetwork.latitude</th>
      <th>geoNetwork.longitude</th>
      <th>geoNetwork.metro</th>
      <th>geoNetwork.networkDomain</th>
      <th>geoNetwork.networkLocation</th>
      <th>geoNetwork.region</th>
      <th>geoNetwork.subContinent</th>
      <th>totals.bounces</th>
      <th>totals.hits</th>
      <th>totals.newVisits</th>
      <th>totals.pageviews</th>
      <th>totals.sessionQualityDim</th>
      <th>totals.timeOnSite</th>
      <th>totals.totalTransactionRevenue</th>
      <th>totals.transactionRevenue</th>
      <th>totals.transactions</th>
      <th>totals.visits</th>
      <th>trafficSource.adContent</th>
      <th>trafficSource.adwordsClickInfo.adNetworkType</th>
      <th>trafficSource.adwordsClickInfo.criteriaParameters</th>
      <th>trafficSource.adwordsClickInfo.gclId</th>
      <th>trafficSource.adwordsClickInfo.isVideoAd</th>
      <th>trafficSource.adwordsClickInfo.page</th>
      <th>trafficSource.adwordsClickInfo.slot</th>
      <th>trafficSource.campaign</th>
      <th>trafficSource.isTrueDirect</th>
      <th>trafficSource.keyword</th>
      <th>trafficSource.medium</th>
      <th>trafficSource.referralPath</th>
      <th>trafficSource.source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>2017-10-16</td>
      <td>3162355547410993243</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '17',...</td>
      <td>Not Socially Engaged</td>
      <td>1508198450</td>
      <td>1</td>
      <td>1508198450</td>
      <td>Firefox</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Europe</td>
      <td>Germany</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>(not set)</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>water bottle</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Referral</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>2017-10-16</td>
      <td>8934116514970143966</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '10',...</td>
      <td>Not Socially Engaged</td>
      <td>1508176307</td>
      <td>6</td>
      <td>1508176307</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Chrome OS</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Cupertino</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>United States</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>San Francisco-Oakland-San Jose CA</td>
      <td>(not set)</td>
      <td>not available in demo dataset</td>
      <td>California</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>NaN</td>
      <td>referral</td>
      <td>/a/google.com/transportation/mtv-services/bike...</td>
      <td>sites.google.com</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Direct</td>
      <td>[{'index': '4', 'value': 'North America'}]</td>
      <td>2017-10-16</td>
      <td>7992466427990357681</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '17',...</td>
      <td>Not Socially Engaged</td>
      <td>1508201613</td>
      <td>1</td>
      <td>1508201613</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>mobile</td>
      <td>not available in demo dataset</td>
      <td>True</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Android</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>United States</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>windjammercable.net</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>True</td>
      <td>NaN</td>
      <td>(none)</td>
      <td>NaN</td>
      <td>(direct)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'EMEA'}]</td>
      <td>2017-10-16</td>
      <td>9075655783635761930</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '9', ...</td>
      <td>Not Socially Engaged</td>
      <td>1508169851</td>
      <td>1</td>
      <td>1508169851</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Asia</td>
      <td>Turkey</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>unknown.unknown</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Western Asia</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Organic Search</td>
      <td>[{'index': '4', 'value': 'Central America'}]</td>
      <td>2017-10-16</td>
      <td>6960673291025684308</td>
      <td>[{'hitNumber': '1', 'time': '0', 'hour': '14',...</td>
      <td>Not Socially Engaged</td>
      <td>1508190552</td>
      <td>1</td>
      <td>1508190552</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>Mexico</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>prod-infinitum.com.mx</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Central America</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
  </tbody>
</table>
</div>



**Data Exploration**

At first, drop some useless columns that has only one variables like 'socialEngagementType'

The number of unique variable in 'socialEngagementType' is 1, as below.


```python
train['socialEngagementType'].nunique(dropna=False)
```




    1




```python
cols_to_drop = [col for col in train.columns if train[col].nunique(dropna=False) == 1]
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop([col for col in cols_to_drop if col in test.columns], axis=1, inplace=True)

print(f'Dropped {len(cols_to_drop)} columns.')
```

    Dropped 19 columns.
    

Change some variables into float data


```python
for col in ['visitNumber', 'totals.hits', 'totals.pageviews', 'totals.transactionRevenue']:
    train[col] = train[col].astype(float)
```

Drop some features


```python
train.drop(['customDimensions', 'hits', 'trafficSource.referralPath', 'trafficSource.source', 'totals.totalTransactionRevenue'], axis=1, inplace=True)
test.drop(['customDimensions', 'hits', 'trafficSource.referralPath', 'trafficSource.source', 'totals.totalTransactionRevenue'], axis=1, inplace=True)
```

This data seems moe simple and looks better.


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channelGrouping</th>
      <th>date</th>
      <th>fullVisitorId</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device.browser</th>
      <th>device.deviceCategory</th>
      <th>device.isMobile</th>
      <th>device.operatingSystem</th>
      <th>geoNetwork.city</th>
      <th>geoNetwork.continent</th>
      <th>geoNetwork.country</th>
      <th>geoNetwork.metro</th>
      <th>geoNetwork.networkDomain</th>
      <th>geoNetwork.region</th>
      <th>geoNetwork.subContinent</th>
      <th>totals.bounces</th>
      <th>totals.hits</th>
      <th>totals.newVisits</th>
      <th>totals.pageviews</th>
      <th>totals.sessionQualityDim</th>
      <th>totals.timeOnSite</th>
      <th>totals.transactionRevenue</th>
      <th>totals.transactions</th>
      <th>trafficSource.adContent</th>
      <th>trafficSource.adwordsClickInfo.adNetworkType</th>
      <th>trafficSource.adwordsClickInfo.gclId</th>
      <th>trafficSource.adwordsClickInfo.isVideoAd</th>
      <th>trafficSource.adwordsClickInfo.page</th>
      <th>trafficSource.adwordsClickInfo.slot</th>
      <th>trafficSource.campaign</th>
      <th>trafficSource.isTrueDirect</th>
      <th>trafficSource.keyword</th>
      <th>trafficSource.medium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>2017-10-16</td>
      <td>3162355547410993243</td>
      <td>1508198450</td>
      <td>1.0</td>
      <td>1508198450</td>
      <td>Firefox</td>
      <td>desktop</td>
      <td>False</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>Europe</td>
      <td>Germany</td>
      <td>not available in demo dataset</td>
      <td>(not set)</td>
      <td>not available in demo dataset</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>water bottle</td>
      <td>organic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Referral</td>
      <td>2017-10-16</td>
      <td>8934116514970143966</td>
      <td>1508176307</td>
      <td>6.0</td>
      <td>1508176307</td>
      <td>Chrome</td>
      <td>desktop</td>
      <td>False</td>
      <td>Chrome OS</td>
      <td>Cupertino</td>
      <td>Americas</td>
      <td>United States</td>
      <td>San Francisco-Oakland-San Jose CA</td>
      <td>(not set)</td>
      <td>California</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>2</td>
      <td>28</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>NaN</td>
      <td>referral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Direct</td>
      <td>2017-10-16</td>
      <td>7992466427990357681</td>
      <td>1508201613</td>
      <td>1.0</td>
      <td>1508201613</td>
      <td>Chrome</td>
      <td>mobile</td>
      <td>True</td>
      <td>Android</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>United States</td>
      <td>not available in demo dataset</td>
      <td>windjammercable.net</td>
      <td>not available in demo dataset</td>
      <td>Northern America</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1</td>
      <td>38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>True</td>
      <td>NaN</td>
      <td>(none)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Organic Search</td>
      <td>2017-10-16</td>
      <td>9075655783635761930</td>
      <td>1508169851</td>
      <td>1.0</td>
      <td>1508169851</td>
      <td>Chrome</td>
      <td>desktop</td>
      <td>False</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>Asia</td>
      <td>Turkey</td>
      <td>not available in demo dataset</td>
      <td>unknown.unknown</td>
      <td>not available in demo dataset</td>
      <td>Western Asia</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>(not provided)</td>
      <td>organic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Organic Search</td>
      <td>2017-10-16</td>
      <td>6960673291025684308</td>
      <td>1508190552</td>
      <td>1.0</td>
      <td>1508190552</td>
      <td>Chrome</td>
      <td>desktop</td>
      <td>False</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>Americas</td>
      <td>Mexico</td>
      <td>not available in demo dataset</td>
      <td>prod-infinitum.com.mx</td>
      <td>not available in demo dataset</td>
      <td>Central America</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>False</td>
      <td>(not provided)</td>
      <td>organic</td>
    </tr>
  </tbody>
</table>
</div>



Let's see the revenue of each visitor.


```python
gdf = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
gdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullVisitorId</th>
      <th>totals.transactionRevenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000245437374675368</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000593255797039768</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000750929315523353</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0001191766179392657</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0003631840334189025</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0004374401845204055</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18381</th>
      <td>99976789209401933</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18382</th>
      <td>99980903580581121</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18383</th>
      <td>9999250019952621738</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>18384 rows × 2 columns</p>
</div>



Make a graph with the table above, sorting with logarithm of revenue.
The graph shows very low percent of visitors make almost revenue.


```python
plt.figure(figsize=(8,6))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()
```


![png](https://github.com/puhuk/puhuk.github.io/blob/master/img/GA-TransactionRev.png?raw=true)



```python
pd.notnull(train["totals.transactionRevenue"])
```




    0        False
    1        False
    2        False
    3        False
             ...  
    19997    False
    19998    False
    19999    False
    Name: totals.transactionRevenue, Length: 20000, dtype: bool



After calculating the ratio of real users, it shows only 1.03% of users make revenue through transaction


```python
print("Number of unique customers with non-zero revenue : ", (gdf["totals.transactionRevenue"]>0).sum(), "and the ratio is : ", round(100*(gdf["totals.transactionRevenue"]>0).sum() / gdf.shape[0],2),"%")
```

    Number of unique customers with non-zero revenue :  189 and the ratio is :  1.03 %
    

Let's look at how many users visit how many times.
It shows about 92% are unique users and only 8% users are revisited.


```python
print("Number of unique visitors in train set : ",train.fullVisitorId.nunique(), " the ratio is : ",round(100*train.fullVisitorId.nunique()/train.shape[0],2),"%")
```

    Number of unique visitors in train set :  18384  the ratio is :  91.92 %
    

Let's explore data by device.
For example, groupping the train data by "browser" with size, count and mean, then the table is as below.


```python
train.groupby('device.browser')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>device.browser</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADM</th>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Amazon Silk</th>
      <td>20</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Android Browser</th>
      <td>11</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Android Webview</th>
      <td>300</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BlackBerry</th>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Chrome</th>
      <td>13978</td>
      <td>171</td>
      <td>1.167348e+08</td>
    </tr>
    <tr>
      <th>Coc Coc</th>
      <td>14</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Edge</th>
      <td>267</td>
      <td>3</td>
      <td>1.348333e+07</td>
    </tr>
    <tr>
      <th>Firefox</th>
      <td>829</td>
      <td>2</td>
      <td>1.349000e+07</td>
    </tr>
    <tr>
      <th>Internet Explorer</th>
      <td>530</td>
      <td>1</td>
      <td>4.200000e+07</td>
    </tr>
    <tr>
      <th>MRCHROME</th>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Maxthon</th>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Mozilla Compatible Agent</th>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Nintendo Browser</th>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Nokia Browser</th>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Opera</th>
      <td>134</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Opera Mini</th>
      <td>191</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Playstation Vita Browser</th>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Puffin</th>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Safari</th>
      <td>3340</td>
      <td>15</td>
      <td>6.022200e+07</td>
    </tr>
    <tr>
      <th>Safari (in-app)</th>
      <td>139</td>
      <td>1</td>
      <td>1.697000e+07</td>
    </tr>
    <tr>
      <th>Samsung Internet</th>
      <td>137</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SeaMonkey</th>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>UC Browser</th>
      <td>48</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>YaBrowser</th>
      <td>36</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Using same logic to browser, device and OS, draw graphs as below.


```python
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

def horizontal_bar_chart(cnt_srs, color):
    trace = go.Bar(
        y=cnt_srs.index[::-1],
        x=cnt_srs.values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

# Device Browser
cnt_srs = train.groupby('device.browser')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Device Category
cnt_srs = train.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace4 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(71, 58, 131, 0.8)')
trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(71, 58, 131, 0.8)')
trace6 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(71, 58, 131, 0.8)')

# Operating system
cnt_srs = train.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace7 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(246, 78, 139, 0.6)')
trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10),'rgba(246, 78, 139, 0.6)')
trace9 = horizontal_bar_chart(cnt_srs["mean"].head(10),'rgba(246, 78, 139, 0.6)')

# Creating two subplots
fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.04, 
                          subplot_titles=["Device Browser - Count", "Device Browser - Non-zero Revenue Count", "Device Browser - Mean Revenue",
                                          "Device Category - Count",  "Device Category - Non-zero Revenue Count", "Device Category - Mean Revenue", 
                                          "Device OS - Count", "Device OS - Non-zero Revenue Count", "Device OS - Mean Revenue"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots')
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>


    This is the format of your plot grid:
    [ (1,1) x1,y1 ]  [ (1,2) x2,y2 ]  [ (1,3) x3,y3 ]
    [ (2,1) x4,y4 ]  [ (2,2) x5,y5 ]  [ (2,3) x6,y6 ]
    [ (3,1) x7,y7 ]  [ (3,2) x8,y8 ]  [ (3,3) x9,y9 ]


![png](https://github.com/puhuk/puhuk.github.io/blob/master/img/GA-Device.png?raw=true)


Below graphs show the continent dependancy.


```python
# Continent
cnt_srs = train.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(58, 71, 80, 0.6)')
trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(58, 71, 80, 0.6)')
trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(58, 71, 80, 0.6)')

# Creating subplots
fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 
                          subplot_titles=["Continent - Count", "Continent - Non-zero Revenue Count", "Continent - Mean Revenue"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=500, width=1200, paper_bgcolor='rgb(233,233,233)', title="Geography Plots")
py.iplot(fig, filename='geo-plots')
```

    This is the format of your plot grid:
    [ (1,1) x1,y1 ]  [ (1,2) x2,y2 ]  [ (1,3) x3,y3 ]


![png](https://github.com/puhuk/puhuk.github.io/blob/master/img/GA-Continent.png?raw=true)


Until now is the part of exploratory data analysis for GA revenue prediction, which I learned from [SRK](https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue) and Andrew [Lukyanenko](https://www.kaggle.com/artgor/fork-of-eda-on-basic-data-and-lgb-in-progress) 's kernel
