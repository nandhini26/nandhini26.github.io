# Online News popularity Analysis


```python
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import matplotlib.pyplot as plt
import seaborn as sns
```

# Reading the source file


```python
data = pd.read_csv("./intermediate_files/OnlineNewsPopularity.csv")
data.shape
pd.options.display.max_columns=61
data.head(3)
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
      <th>url</th>
      <th>timedelta</th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>data_channel_is_lifestyle</th>
      <th>data_channel_is_entertainment</th>
      <th>data_channel_is_bus</th>
      <th>data_channel_is_socmed</th>
      <th>data_channel_is_tech</th>
      <th>data_channel_is_world</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>weekday_is_monday</th>
      <th>weekday_is_tuesday</th>
      <th>weekday_is_wednesday</th>
      <th>weekday_is_thursday</th>
      <th>weekday_is_friday</th>
      <th>weekday_is_saturday</th>
      <th>weekday_is_sunday</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>http://mashable.com/2013/01/07/amazon-instant-...</td>
      <td>731.0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.680365</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500331</td>
      <td>0.378279</td>
      <td>0.040005</td>
      <td>0.041263</td>
      <td>0.040123</td>
      <td>0.521617</td>
      <td>0.092562</td>
      <td>0.045662</td>
      <td>0.013699</td>
      <td>0.769231</td>
      <td>0.230769</td>
      <td>0.378636</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600</td>
      <td>-0.200000</td>
      <td>0.5</td>
      <td>-0.1875</td>
      <td>0.0</td>
      <td>0.1875</td>
      <td>593</td>
    </tr>
    <tr>
      <td>1</td>
      <td>http://mashable.com/2013/01/07/ap-samsung-spon...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>255.0</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.913725</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.799756</td>
      <td>0.050047</td>
      <td>0.050096</td>
      <td>0.050101</td>
      <td>0.050001</td>
      <td>0.341246</td>
      <td>0.148948</td>
      <td>0.043137</td>
      <td>0.015686</td>
      <td>0.733333</td>
      <td>0.266667</td>
      <td>0.286915</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125</td>
      <td>-0.100000</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>711</td>
    </tr>
    <tr>
      <td>2</td>
      <td>http://mashable.com/2013/01/07/apple-40-billio...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>211.0</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.393365</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.217792</td>
      <td>0.033334</td>
      <td>0.033351</td>
      <td>0.033334</td>
      <td>0.682188</td>
      <td>0.702222</td>
      <td>0.323333</td>
      <td>0.056872</td>
      <td>0.009479</td>
      <td>0.857143</td>
      <td>0.142857</td>
      <td>0.495833</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800</td>
      <td>-0.133333</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>1500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merging the weekdays columns channels as one single column
publishday=data[[' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday', 
                      ' weekday_is_thursday', ' weekday_is_friday',' weekday_is_saturday' ,' weekday_is_sunday' ]]
```


```python
publishday.head(3)
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
      <th>weekday_is_monday</th>
      <th>weekday_is_tuesday</th>
      <th>weekday_is_wednesday</th>
      <th>weekday_is_thursday</th>
      <th>weekday_is_friday</th>
      <th>weekday_is_saturday</th>
      <th>weekday_is_sunday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reducing the columns into one column
day_arr=[]
for r in list(range(publishday.shape[0])):# row 
    for c in list(range(publishday.shape[1])):# column
        if ((c==0) and (publishday.iloc[r,c])==1):
            day_arr.append('Monday')
        elif ((c==1) and (publishday.iloc[r,c])==1):
            day_arr.append('Tueday')
        elif ((c==2) and (publishday.iloc[r,c])==1):
            day_arr.append('Wednesday')
        elif ((c==3) and (publishday.iloc[r,c])==1):
            day_arr.append('Thursday')
        elif ((c==4) and (publishday.iloc[r,c])==1):
            day_arr.append('Friday')
        elif ((c==5) and (publishday.iloc[r,c])==1):
            day_arr.append('Saturday') 
        elif ((c==6) and (publishday.iloc[r,c])==1):
            day_arr.append('Sunday')
```


```python
# merge the the new data into the dataframe
data.insert(loc=11, column='weekdays', value=day_arr)

```


```python
pd.options.display.max_columns=62
data.shape
```




    (39644, 62)




```python
data.head(1)
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
      <th>url</th>
      <th>timedelta</th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>weekdays</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>data_channel_is_lifestyle</th>
      <th>data_channel_is_entertainment</th>
      <th>data_channel_is_bus</th>
      <th>data_channel_is_socmed</th>
      <th>data_channel_is_tech</th>
      <th>data_channel_is_world</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>weekday_is_monday</th>
      <th>weekday_is_tuesday</th>
      <th>weekday_is_wednesday</th>
      <th>weekday_is_thursday</th>
      <th>weekday_is_friday</th>
      <th>weekday_is_saturday</th>
      <th>weekday_is_sunday</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>http://mashable.com/2013/01/07/amazon-instant-...</td>
      <td>731.0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>4.680365</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500331</td>
      <td>0.378279</td>
      <td>0.040005</td>
      <td>0.041263</td>
      <td>0.040123</td>
      <td>0.521617</td>
      <td>0.092562</td>
      <td>0.045662</td>
      <td>0.013699</td>
      <td>0.769231</td>
      <td>0.230769</td>
      <td>0.378636</td>
      <td>0.1</td>
      <td>0.7</td>
      <td>-0.35</td>
      <td>-0.6</td>
      <td>-0.2</td>
      <td>0.5</td>
      <td>-0.1875</td>
      <td>0.0</td>
      <td>0.1875</td>
      <td>593</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc[data.weekdays=='Monday']
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
      <th>url</th>
      <th>timedelta</th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>weekdays</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>data_channel_is_lifestyle</th>
      <th>data_channel_is_entertainment</th>
      <th>data_channel_is_bus</th>
      <th>data_channel_is_socmed</th>
      <th>data_channel_is_tech</th>
      <th>data_channel_is_world</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>weekday_is_monday</th>
      <th>weekday_is_tuesday</th>
      <th>weekday_is_wednesday</th>
      <th>weekday_is_thursday</th>
      <th>weekday_is_friday</th>
      <th>weekday_is_saturday</th>
      <th>weekday_is_sunday</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>http://mashable.com/2013/01/07/amazon-instant-...</td>
      <td>731.0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>4.680365</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>496.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500331</td>
      <td>0.378279</td>
      <td>0.040005</td>
      <td>0.041263</td>
      <td>0.040123</td>
      <td>0.521617</td>
      <td>0.092562</td>
      <td>0.045662</td>
      <td>0.013699</td>
      <td>0.769231</td>
      <td>0.230769</td>
      <td>0.378636</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600000</td>
      <td>-0.200000</td>
      <td>0.500000</td>
      <td>-0.187500</td>
      <td>0.000000</td>
      <td>0.187500</td>
      <td>593</td>
    </tr>
    <tr>
      <td>1</td>
      <td>http://mashable.com/2013/01/07/ap-samsung-spon...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>255.0</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>4.913725</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.799756</td>
      <td>0.050047</td>
      <td>0.050096</td>
      <td>0.050101</td>
      <td>0.050001</td>
      <td>0.341246</td>
      <td>0.148948</td>
      <td>0.043137</td>
      <td>0.015686</td>
      <td>0.733333</td>
      <td>0.266667</td>
      <td>0.286915</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125000</td>
      <td>-0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>711</td>
    </tr>
    <tr>
      <td>2</td>
      <td>http://mashable.com/2013/01/07/apple-40-billio...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>211.0</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>4.393365</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>918.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.217792</td>
      <td>0.033334</td>
      <td>0.033351</td>
      <td>0.033334</td>
      <td>0.682188</td>
      <td>0.702222</td>
      <td>0.323333</td>
      <td>0.056872</td>
      <td>0.009479</td>
      <td>0.857143</td>
      <td>0.142857</td>
      <td>0.495833</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800000</td>
      <td>-0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1500</td>
    </tr>
    <tr>
      <td>3</td>
      <td>http://mashable.com/2013/01/07/astronaut-notre...</td>
      <td>731.0</td>
      <td>9.0</td>
      <td>531.0</td>
      <td>0.503788</td>
      <td>1.0</td>
      <td>0.665635</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>4.404896</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.028573</td>
      <td>0.419300</td>
      <td>0.494651</td>
      <td>0.028905</td>
      <td>0.028572</td>
      <td>0.429850</td>
      <td>0.100705</td>
      <td>0.041431</td>
      <td>0.020716</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.385965</td>
      <td>0.136364</td>
      <td>0.8</td>
      <td>-0.369697</td>
      <td>-0.600000</td>
      <td>-0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1200</td>
    </tr>
    <tr>
      <td>4</td>
      <td>http://mashable.com/2013/01/07/att-u-verse-apps/</td>
      <td>731.0</td>
      <td>13.0</td>
      <td>1072.0</td>
      <td>0.415646</td>
      <td>1.0</td>
      <td>0.540890</td>
      <td>19.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>4.682836</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>545.0</td>
      <td>16000.0</td>
      <td>3151.157895</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.028633</td>
      <td>0.028794</td>
      <td>0.028575</td>
      <td>0.028572</td>
      <td>0.885427</td>
      <td>0.513502</td>
      <td>0.281003</td>
      <td>0.074627</td>
      <td>0.012127</td>
      <td>0.860215</td>
      <td>0.139785</td>
      <td>0.411127</td>
      <td>0.033333</td>
      <td>1.0</td>
      <td>-0.220192</td>
      <td>-0.500000</td>
      <td>-0.050000</td>
      <td>0.454545</td>
      <td>0.136364</td>
      <td>0.045455</td>
      <td>0.136364</td>
      <td>505</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>39574</td>
      <td>http://mashable.com/2014/12/25/romanian-childr...</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>367.0</td>
      <td>0.534819</td>
      <td>1.0</td>
      <td>0.647059</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Monday</td>
      <td>4.354223</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1300.0</td>
      <td>481.428571</td>
      <td>1300.0</td>
      <td>843300.0</td>
      <td>323742.857143</td>
      <td>1229.431818</td>
      <td>4301.954315</td>
      <td>2557.588194</td>
      <td>957.0</td>
      <td>1100.0</td>
      <td>1028.500000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.885193</td>
      <td>0.028573</td>
      <td>0.028785</td>
      <td>0.028874</td>
      <td>0.028575</td>
      <td>0.523965</td>
      <td>0.013005</td>
      <td>0.038147</td>
      <td>0.019074</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.223701</td>
      <td>0.136364</td>
      <td>0.5</td>
      <td>-0.363095</td>
      <td>-0.500000</td>
      <td>-0.125000</td>
      <td>0.454545</td>
      <td>0.136364</td>
      <td>0.045455</td>
      <td>0.136364</td>
      <td>1400</td>
    </tr>
    <tr>
      <td>39575</td>
      <td>http://mashable.com/2014/12/25/samsung-galaxy-...</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>321.0</td>
      <td>0.619048</td>
      <td>1.0</td>
      <td>0.691542</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>4.940810</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>616.0</td>
      <td>96.666667</td>
      <td>6500.0</td>
      <td>843300.0</td>
      <td>249111.111111</td>
      <td>1663.618182</td>
      <td>5395.403976</td>
      <td>3464.006341</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>500.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.022800</td>
      <td>0.022227</td>
      <td>0.161639</td>
      <td>0.631223</td>
      <td>0.162112</td>
      <td>0.645833</td>
      <td>0.054213</td>
      <td>0.037383</td>
      <td>0.015576</td>
      <td>0.705882</td>
      <td>0.294118</td>
      <td>0.396528</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.640000</td>
      <td>-1.000000</td>
      <td>-0.100000</td>
      <td>0.500000</td>
      <td>-0.350000</td>
      <td>0.000000</td>
      <td>0.350000</td>
      <td>1300</td>
    </tr>
    <tr>
      <td>39576</td>
      <td>http://mashable.com/2014/12/25/selfie-stick-id...</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>454.0</td>
      <td>0.607477</td>
      <td>1.0</td>
      <td>0.709459</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Monday</td>
      <td>4.997797</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>451.0</td>
      <td>103.750000</td>
      <td>0.0</td>
      <td>843300.0</td>
      <td>186211.111111</td>
      <td>0.000000</td>
      <td>5123.722892</td>
      <td>2549.749578</td>
      <td>2200.0</td>
      <td>2200.0</td>
      <td>2200.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.022230</td>
      <td>0.022225</td>
      <td>0.646937</td>
      <td>0.022264</td>
      <td>0.286344</td>
      <td>0.403340</td>
      <td>-0.104912</td>
      <td>0.013216</td>
      <td>0.041850</td>
      <td>0.240000</td>
      <td>0.760000</td>
      <td>0.395455</td>
      <td>0.100000</td>
      <td>0.8</td>
      <td>-0.230848</td>
      <td>-0.625000</td>
      <td>-0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1500</td>
    </tr>
    <tr>
      <td>39577</td>
      <td>http://mashable.com/2014/12/25/snowstorm-mosco...</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>363.0</td>
      <td>0.594828</td>
      <td>1.0</td>
      <td>0.710526</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>5.181818</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>699.0</td>
      <td>184.655714</td>
      <td>699.0</td>
      <td>843300.0</td>
      <td>227242.714286</td>
      <td>699.000000</td>
      <td>5021.962727</td>
      <td>2698.160007</td>
      <td>568.0</td>
      <td>568.0</td>
      <td>568.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.028747</td>
      <td>0.028575</td>
      <td>0.451610</td>
      <td>0.028596</td>
      <td>0.462473</td>
      <td>0.267778</td>
      <td>0.146421</td>
      <td>0.022039</td>
      <td>0.005510</td>
      <td>0.800000</td>
      <td>0.200000</td>
      <td>0.406331</td>
      <td>0.100000</td>
      <td>0.8</td>
      <td>-0.161111</td>
      <td>-0.166667</td>
      <td>-0.155556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>642</td>
    </tr>
    <tr>
      <td>39578</td>
      <td>http://mashable.com/2014/12/25/teen-pulled-ove...</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>287.0</td>
      <td>0.602151</td>
      <td>1.0</td>
      <td>0.748663</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Monday</td>
      <td>5.038328</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1200.0</td>
      <td>268.142857</td>
      <td>1900.0</td>
      <td>843300.0</td>
      <td>341000.000000</td>
      <td>1380.393443</td>
      <td>3386.493489</td>
      <td>2540.014750</td>
      <td>1100.0</td>
      <td>2700.0</td>
      <td>1775.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.457856</td>
      <td>0.028573</td>
      <td>0.028572</td>
      <td>0.028572</td>
      <td>0.456426</td>
      <td>0.495661</td>
      <td>0.122107</td>
      <td>0.041812</td>
      <td>0.017422</td>
      <td>0.705882</td>
      <td>0.294118</td>
      <td>0.408864</td>
      <td>0.136364</td>
      <td>1.0</td>
      <td>-0.380000</td>
      <td>-1.000000</td>
      <td>-0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>2000</td>
    </tr>
  </tbody>
</table>
<p>6661 rows × 62 columns</p>
</div>




```python
# Merging the data channels as one single column
DataChannel=data[[' data_channel_is_lifestyle',' data_channel_is_entertainment' ,' data_channel_is_bus',
                        ' data_channel_is_socmed' ,' data_channel_is_tech',' data_channel_is_world' ]]
```


```python
Data_channel=[]

for r in list(range(DataChannel.shape[0])):# row 
    if (((DataChannel.iloc[r,0])==0) and ((DataChannel.iloc[r,1])==0) and ((DataChannel.iloc[r,2])==0) and ((DataChannel.iloc[r,3])==0) and ((DataChannel.iloc[r,4])==0) and ((DataChannel.iloc[r,5])==0)):
        Data_channel.append('Others')
    for c in list(range(DataChannel.shape[1])):# column
        if ((c==0) and (DataChannel.iloc[r,c])==1):            
            Data_channel.append('Life')
        elif ((c==1) and (DataChannel.iloc[r,c])==1):
            Data_channel.append('Enter')
        elif ((c==2) and (DataChannel.iloc[r,c])==1):
            Data_channel.append('Busi')
        elif ((c==3) and (DataChannel.iloc[r,c])==1):
            Data_channel.append('Somed')
        elif ((c==4) and (DataChannel.iloc[r,c])==1):
            Data_channel.append('Tech')
        elif ((c==5) and (DataChannel.iloc[r,c])==1):
            Data_channel.append('World')       
```


```python
my_df = pd.DataFrame(Data_channel)
```


```python
my_df.shape
```




    (39644, 1)




```python
# merge the the new data into the dataframe

data.insert(loc=12, column='data_channel', value=Data_channel)
```


```python
# Now I drop the old data
data.drop(labels=[' data_channel_is_lifestyle',' data_channel_is_entertainment' ,' data_channel_is_bus',
                        ' data_channel_is_socmed' ,' data_channel_is_tech',' data_channel_is_world', 
                 ' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday', 
                      ' weekday_is_thursday', ' weekday_is_friday',' weekday_is_saturday' ,' weekday_is_sunday'], axis = 1, inplace=True)
```

#  Total number of column has reduced from 62 to 50


```python
data.shape
```




    (39644, 50)




```python
# Here we drop the two non-preditive (url and timedelta) attributes. They won't contribute anything
data.drop(labels=['url',' timedelta'], axis = 1, inplace=True)
```


```python
data.shape
```




    (39644, 48)




```python
# creating a grading criteria for the shares
share_data = data[' shares']
data[' shares'].describe()
```




    count     39644.000000
    mean       3395.380184
    std       11626.950749
    min           1.000000
    25%         946.000000
    50%        1400.000000
    75%        2800.000000
    max      843300.000000
    Name:  shares, dtype: float64



# Categorizing the shares into different popolarity


```python
# create label grades for the classes
share_label = list()
for share in share_data:
    if share <= 645:
        share_label.append('Very Poor')
    elif share > 645 and share <= 861:
        share_label.append('Poor')
    elif share > 861 and share <= 1400:
        share_label.append('Average')
    elif share > 1400 and share <= 31300:
        share_label.append('Good')
    elif share > 31300 and share <= 53700:
        share_label.append('Very Good')
    elif share > 53700 and share <= 77200:
        share_label.append('Excellent')
    else:
        share_label.append('Exceptional')

# Update this class label into the dataframe
data = pd.concat([data, pd.DataFrame(share_label, columns=['popularity'])], axis=1)
data.head(4)
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
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>weekdays</th>
      <th>data_channel</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Enter</td>
      <td>4.680365</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>0.0</td>
      <td>0.500331</td>
      <td>0.378279</td>
      <td>0.040005</td>
      <td>0.041263</td>
      <td>0.040123</td>
      <td>0.521617</td>
      <td>0.092562</td>
      <td>0.045662</td>
      <td>0.013699</td>
      <td>0.769231</td>
      <td>0.230769</td>
      <td>0.378636</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600</td>
      <td>-0.200000</td>
      <td>0.5</td>
      <td>-0.1875</td>
      <td>0.0</td>
      <td>0.1875</td>
      <td>593</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9.0</td>
      <td>255.0</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Busi</td>
      <td>4.913725</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.799756</td>
      <td>0.050047</td>
      <td>0.050096</td>
      <td>0.050101</td>
      <td>0.050001</td>
      <td>0.341246</td>
      <td>0.148948</td>
      <td>0.043137</td>
      <td>0.015686</td>
      <td>0.733333</td>
      <td>0.266667</td>
      <td>0.286915</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125</td>
      <td>-0.100000</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>711</td>
      <td>Poor</td>
    </tr>
    <tr>
      <td>2</td>
      <td>9.0</td>
      <td>211.0</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Busi</td>
      <td>4.393365</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>0.0</td>
      <td>0.217792</td>
      <td>0.033334</td>
      <td>0.033351</td>
      <td>0.033334</td>
      <td>0.682188</td>
      <td>0.702222</td>
      <td>0.323333</td>
      <td>0.056872</td>
      <td>0.009479</td>
      <td>0.857143</td>
      <td>0.142857</td>
      <td>0.495833</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800</td>
      <td>-0.133333</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>1500</td>
      <td>Good</td>
    </tr>
    <tr>
      <td>3</td>
      <td>9.0</td>
      <td>531.0</td>
      <td>0.503788</td>
      <td>1.0</td>
      <td>0.665635</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Enter</td>
      <td>4.404896</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.028573</td>
      <td>0.419300</td>
      <td>0.494651</td>
      <td>0.028905</td>
      <td>0.028572</td>
      <td>0.429850</td>
      <td>0.100705</td>
      <td>0.041431</td>
      <td>0.020716</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.385965</td>
      <td>0.136364</td>
      <td>0.8</td>
      <td>-0.369697</td>
      <td>-0.600</td>
      <td>-0.166667</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>1200</td>
      <td>Average</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Evaluating features (sensors) contribution towards the label
fig = plt.figure(figsize=(15,5))
ax = sns.countplot(x='popularity',hue='data_channel',data=data,alpha=0.5)
```


![png](/images/output_24_0.png)



```python
# remove noise from n_tokens_content. those equals to 0

data  = data[data[' n_tokens_content'] != 0]
```


```python
data.head(3)
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
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>weekdays</th>
      <th>data_channel</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
      <th>popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Enter</td>
      <td>4.680365</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>0.0</td>
      <td>0.500331</td>
      <td>0.378279</td>
      <td>0.040005</td>
      <td>0.041263</td>
      <td>0.040123</td>
      <td>0.521617</td>
      <td>0.092562</td>
      <td>0.045662</td>
      <td>0.013699</td>
      <td>0.769231</td>
      <td>0.230769</td>
      <td>0.378636</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600</td>
      <td>-0.200000</td>
      <td>0.5</td>
      <td>-0.1875</td>
      <td>0.0</td>
      <td>0.1875</td>
      <td>593</td>
      <td>Very Poor</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9.0</td>
      <td>255.0</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Busi</td>
      <td>4.913725</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.799756</td>
      <td>0.050047</td>
      <td>0.050096</td>
      <td>0.050101</td>
      <td>0.050001</td>
      <td>0.341246</td>
      <td>0.148948</td>
      <td>0.043137</td>
      <td>0.015686</td>
      <td>0.733333</td>
      <td>0.266667</td>
      <td>0.286915</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125</td>
      <td>-0.100000</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>711</td>
      <td>Poor</td>
    </tr>
    <tr>
      <td>2</td>
      <td>9.0</td>
      <td>211.0</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Busi</td>
      <td>4.393365</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>0.0</td>
      <td>0.217792</td>
      <td>0.033334</td>
      <td>0.033351</td>
      <td>0.033334</td>
      <td>0.682188</td>
      <td>0.702222</td>
      <td>0.323333</td>
      <td>0.056872</td>
      <td>0.009479</td>
      <td>0.857143</td>
      <td>0.142857</td>
      <td>0.495833</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800</td>
      <td>-0.133333</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>1500</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (38463, 49)




```python
# describing the data
data.describe()
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
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>10.382419</td>
      <td>563.295375</td>
      <td>0.565049</td>
      <td>1.027065</td>
      <td>0.710336</td>
      <td>11.217872</td>
      <td>3.394769</td>
      <td>4.563061</td>
      <td>1.263786</td>
      <td>4.687892</td>
      <td>7.215012</td>
      <td>26.708187</td>
      <td>1151.751079</td>
      <td>313.946906</td>
      <td>13182.545563</td>
      <td>750317.505135</td>
      <td>255215.159411</td>
      <td>1102.009897</td>
      <td>5603.782810</td>
      <td>3103.427793</td>
      <td>4121.536513</td>
      <td>10646.369414</td>
      <td>6598.260636</td>
      <td>0.130671</td>
      <td>0.188134</td>
      <td>0.141680</td>
      <td>0.217177</td>
      <td>0.214291</td>
      <td>0.238692</td>
      <td>0.456984</td>
      <td>0.122973</td>
      <td>0.040842</td>
      <td>0.017122</td>
      <td>0.703096</td>
      <td>0.296774</td>
      <td>0.364689</td>
      <td>0.098376</td>
      <td>0.779963</td>
      <td>-0.267493</td>
      <td>-0.537970</td>
      <td>-0.110801</td>
      <td>0.280573</td>
      <td>0.070997</td>
      <td>0.342431</td>
      <td>0.154930</td>
      <td>3355.360398</td>
    </tr>
    <tr>
      <td>std</td>
      <td>2.113800</td>
      <td>468.299538</td>
      <td>3.573022</td>
      <td>5.307978</td>
      <td>3.312293</td>
      <td>11.340580</td>
      <td>3.869773</td>
      <td>8.295365</td>
      <td>4.164896</td>
      <td>0.283231</td>
      <td>1.916459</td>
      <td>70.278215</td>
      <td>3870.494630</td>
      <td>624.449580</td>
      <td>56850.480221</td>
      <td>216395.888328</td>
      <td>131821.240188</td>
      <td>1127.031740</td>
      <td>6096.725673</td>
      <td>1301.238777</td>
      <td>20026.792850</td>
      <td>41612.149801</td>
      <td>24553.836601</td>
      <td>0.337045</td>
      <td>0.265547</td>
      <td>0.220360</td>
      <td>0.282238</td>
      <td>0.288938</td>
      <td>0.291382</td>
      <td>0.088386</td>
      <td>0.096091</td>
      <td>0.016229</td>
      <td>0.010588</td>
      <td>0.150208</td>
      <td>0.150032</td>
      <td>0.085455</td>
      <td>0.070382</td>
      <td>0.212509</td>
      <td>0.121174</td>
      <td>0.279703</td>
      <td>0.094919</td>
      <td>0.323561</td>
      <td>0.264338</td>
      <td>0.188606</td>
      <td>0.225636</td>
      <td>11585.968776</td>
    </tr>
    <tr>
      <td>min</td>
      <td>2.000000</td>
      <td>18.000000</td>
      <td>0.114964</td>
      <td>1.000000</td>
      <td>0.119134</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.600000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.393750</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>9.000000</td>
      <td>259.000000</td>
      <td>0.477419</td>
      <td>1.000000</td>
      <td>0.632588</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.496250</td>
      <td>6.000000</td>
      <td>-1.000000</td>
      <td>445.000000</td>
      <td>143.000000</td>
      <td>0.000000</td>
      <td>843300.000000</td>
      <td>171300.000000</td>
      <td>0.000000</td>
      <td>3549.290325</td>
      <td>2373.807082</td>
      <td>703.000000</td>
      <td>1200.000000</td>
      <td>1100.000000</td>
      <td>0.000000</td>
      <td>0.025060</td>
      <td>0.025012</td>
      <td>0.028572</td>
      <td>0.025622</td>
      <td>0.028575</td>
      <td>0.402457</td>
      <td>0.064394</td>
      <td>0.029463</td>
      <td>0.010177</td>
      <td>0.612903</td>
      <td>0.200000</td>
      <td>0.311880</td>
      <td>0.050000</td>
      <td>0.600000</td>
      <td>-0.331532</td>
      <td>-0.714286</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>945.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>10.000000</td>
      <td>423.000000</td>
      <td>0.542986</td>
      <td>1.000000</td>
      <td>0.693727</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>4.674121</td>
      <td>7.000000</td>
      <td>-1.000000</td>
      <td>660.000000</td>
      <td>237.600000</td>
      <td>1400.000000</td>
      <td>843300.000000</td>
      <td>242080.000000</td>
      <td>1009.000000</td>
      <td>4311.457071</td>
      <td>2850.846753</td>
      <td>1200.000000</td>
      <td>3000.000000</td>
      <td>2300.000000</td>
      <td>0.000000</td>
      <td>0.033423</td>
      <td>0.033345</td>
      <td>0.040008</td>
      <td>0.040000</td>
      <td>0.050000</td>
      <td>0.456566</td>
      <td>0.122517</td>
      <td>0.039604</td>
      <td>0.015674</td>
      <td>0.714286</td>
      <td>0.285714</td>
      <td>0.361872</td>
      <td>0.100000</td>
      <td>0.800000</td>
      <td>-0.257738</td>
      <td>-0.500000</td>
      <td>-0.100000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1400.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>12.000000</td>
      <td>729.000000</td>
      <td>0.611111</td>
      <td>1.000000</td>
      <td>0.756944</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>4.861901</td>
      <td>9.000000</td>
      <td>4.000000</td>
      <td>1000.000000</td>
      <td>359.200000</td>
      <td>7700.000000</td>
      <td>843300.000000</td>
      <td>326879.464285</td>
      <td>2031.249361</td>
      <td>5962.421633</td>
      <td>3550.518235</td>
      <td>2700.000000</td>
      <td>8200.000000</td>
      <td>5301.535714</td>
      <td>0.000000</td>
      <td>0.251934</td>
      <td>0.150692</td>
      <td>0.335022</td>
      <td>0.340502</td>
      <td>0.414587</td>
      <td>0.510305</td>
      <td>0.179916</td>
      <td>0.050725</td>
      <td>0.021987</td>
      <td>0.800000</td>
      <td>0.387097</td>
      <td>0.413254</td>
      <td>0.100000</td>
      <td>1.000000</td>
      <td>-0.193415</td>
      <td>-0.312500</td>
      <td>-0.050000</td>
      <td>0.500000</td>
      <td>0.136364</td>
      <td>0.500000</td>
      <td>0.250000</td>
      <td>2700.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>23.000000</td>
      <td>8474.000000</td>
      <td>701.000000</td>
      <td>1042.000000</td>
      <td>650.000000</td>
      <td>304.000000</td>
      <td>116.000000</td>
      <td>128.000000</td>
      <td>91.000000</td>
      <td>8.041534</td>
      <td>10.000000</td>
      <td>377.000000</td>
      <td>298400.000000</td>
      <td>42827.857143</td>
      <td>843300.000000</td>
      <td>843300.000000</td>
      <td>843300.000000</td>
      <td>3613.039820</td>
      <td>298400.000000</td>
      <td>43567.659946</td>
      <td>843300.000000</td>
      <td>843300.000000</td>
      <td>843300.000000</td>
      <td>1.000000</td>
      <td>0.926994</td>
      <td>0.925947</td>
      <td>0.919999</td>
      <td>0.926534</td>
      <td>0.927191</td>
      <td>1.000000</td>
      <td>0.727841</td>
      <td>0.155488</td>
      <td>0.184932</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>843300.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fetch the counts for each class
class_counts = data.groupby('popularity').size().reset_index()
class_counts.columns = ['Popularity','No of articles']
class_counts
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
      <th>Popularity</th>
      <th>No of articles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Average</td>
      <td>12064</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Excellent</td>
      <td>75</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Exceptional</td>
      <td>88</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Good</td>
      <td>18522</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Poor</td>
      <td>4723</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Very Good</td>
      <td>227</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Very Poor</td>
      <td>2764</td>
    </tr>
  </tbody>
</table>
</div>




```python
# n_non_stop_unique_tokens
data[' n_non_stop_unique_tokens'].describe()
```




    count    38463.000000
    mean         0.710336
    std          3.312293
    min          0.119134
    25%          0.632588
    50%          0.693727
    75%          0.756944
    max        650.000000
    Name:  n_non_stop_unique_tokens, dtype: float64




```python
data[' shares'].describe()
```




    count     38463.000000
    mean       3355.360398
    std       11585.968776
    min           1.000000
    25%         945.000000
    50%        1400.000000
    75%        2700.000000
    max      843300.000000
    Name:  shares, dtype: float64




```python
temp_data = data[data[' shares'] >= 10000]
fig, axes = plt.subplots(figsize=(10,9))
# box plot
sns.boxplot(x='popularity', y=' n_non_stop_unique_tokens', width=0.3,data=temp_data, ax=axes)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1a015a50>




![png](/images/output_32_1.png)



```python
#kw_min_min and related kw_ terms

temp_data = data[data[' shares'] <= 10000]

# running a pair plot for the kw__terms

kw_cols = [' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg', 
            ' kw_max_avg', ' kw_avg_avg', ' shares']

# run a pairplot

sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='hist')
```




    <seaborn.axisgrid.PairGrid at 0x1a25154510>




![png](/images/output_33_1.png)


***Finding the relationship between the words and shares***


```python

temp_data = data[data[' shares'] <= 100000]

# running a pair plot for the these terms

words_cols = [' rate_positive_words', ' rate_negative_words', ' global_rate_positive_words', ' global_rate_negative_words', ' shares']

# run a pairplot

sns.pairplot(temp_data, vars=words_cols, hue='popularity', diag_kind='kde')

```




    <seaborn.axisgrid.PairGrid at 0x1a32d7dfd0>




![png](/images/output_35_1.png)


# Findings
    There is a negative linear relationship between rate_positive_words and rate_negative_words (it is expected)
    rate_positive_words: Most of articles tends to be on falls towards the 0.3 - 1
    rate_negative_words: Most of articles tends to be on falls towards the 0.8 - 0


```python
# attempt polartiy
temp_data = data[data[' shares'] <= 100000]
sns.lmplot(x=' avg_positive_polarity', y=' shares', col='popularity', data=temp_data)
```




    <seaborn.axisgrid.FacetGrid at 0x1a367a5810>




![png](/images/output_37_1.png)



```python
# attempt polartiy
temp_data = data[data[' shares'] <= 100000]
fig, axes = plt.subplots(figsize=(10,10))
sns.scatterplot(x=' avg_positive_polarity', y=' shares', hue='popularity', data=temp_data, ax=axes)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1d64f550>




![png](/images/output_38_1.png)


# Findings
   Most of the article fall in the good popolarity category


```python
#Finding relationship between 'rate_positive_words', 'rate_negative_words', 'global_rate_positive_words', 'global_rate_negative_words', and 'shares'

temp_data = data[data[' shares'] <= 100000]

# running a pair plot for the terms

kw_cols = [' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity', ' min_negative_polarity', ' max_negative_polarity', ' shares']

# run a pairplot

sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
```




    <seaborn.axisgrid.PairGrid at 0x1a3c40f310>




![png](/images/output_40_1.png)



```python
# Find the sentiment polarity  with shares

temp_data = data[data[' shares'] <= 100000]
fig, axes = plt.subplots(figsize=(10,10))
sns.scatterplot(x=' title_sentiment_polarity', y=' shares', hue='popularity', data=temp_data, ax=axes)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a42bb7290>




![png](/images/output_41_1.png)



```python
# find title_subjectivity with Shares

temp_data = data[data[' shares'] <= 10000]

fig, axes = plt.subplots(figsize=(15,15))

sns.scatterplot(x=' title_subjectivity', y=' shares', hue='popularity', data=temp_data, ax=axes)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a4abda990>




![png](/images/output_42_1.png)



```python
temp_data = data[data[' shares'] <= 100000]

# finding for polarity versus popoularity

kw_cols = [' title_sentiment_polarity', ' abs_title_sentiment_polarity', ' title_subjectivity', ' abs_title_subjectivity', ' shares']

# run a pairplot

sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
```




    <seaborn.axisgrid.PairGrid at 0x1a46f85210>




![png](/images/output_43_1.png)



```python
# attempt self_reference_min_shares
temp_data = data[(data[' shares'] <= 100000) & (data[' self_reference_min_shares'] <= 30000)]
fig, axes = plt.subplots(figsize=(15,15))
sns.scatterplot(x=' self_reference_min_shares', y=' shares', hue= 'popularity', data=temp_data, ax=axes)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a4fc42590>




![png](/images/output_44_1.png)



```python
temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' self_reference_min_shares', ' self_reference_max_shares', ' self_reference_avg_sharess', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
```




    <seaborn.axisgrid.PairGrid at 0x1a4fad7210>




![png](/images/output_45_1.png)



```python
#### LDA - 0: 5
temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
```




    <seaborn.axisgrid.PairGrid at 0x1a4f45ed10>




![png](/images/output_46_1.png)



```python

```

### Weekdays Analysis


```python
# extact the weekdays articles distrubution
weekdays_data = data.groupby('weekdays').size().reset_index()
weekdays_data.columns = ['weekdays','count']
weekdays_data
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
      <th>weekdays</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Friday</td>
      <td>5538</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Monday</td>
      <td>6471</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Saturday</td>
      <td>2369</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Sunday</td>
      <td>2657</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Thursday</td>
      <td>7052</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Tueday</td>
      <td>7171</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Wednesday</td>
      <td>7205</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shows the days when articles are usually posted
fig = plt.subplots(figsize=(10,10))
sns.countplot(x='weekdays',data=data,alpha=0.8)
plt.show()
```


![png](/images/output_50_0.png)



    Findings:
    
    Mid of the week is seem to have large counts of shares than other days. Tuesday,Wednesday and Thursday are the days which has more number of publish.


```python
# shows relationship with the number of shares and the weekdays
temp_data = data[(data['popularity'] == 'Very Poor') | (data['popularity'] == 'Poor') | (data['popularity'] == 'Average') | (data['popularity'] == 'Good')]
ax = sns.catplot(x='weekdays', col="popularity", data=temp_data, kind="count", height=10, aspect=.7)
```


![png](/images/output_52_0.png)


Findings:
    
    Mid of the week is seem to be best day to publish. Tuesday,Wednesday and Thursday are the best day to publish.


```python

temp_data = data[(data['popularity'] == 'Exceptional') | (data['popularity'] == 'Excellent') | (data['popularity'] == 'Very Good')]
ax = sns.catplot(x='weekdays', col="popularity", data=temp_data, kind="count", height=20, aspect=.7)
```


![png](/images/output_54_0.png)


# Findings:

It seems the best popular articles are usually posted on Mondays and Wednesday (and a bit of tuesdays) Sundays and Saturdays (Weekends generally) are the worsts days to publish an articles. Hit ratios would be less.




```python
temp_data = data[data[' shares'] <= 100000]
# running a pair plot for the kw__terms
kw_cols = [' average_token_length', ' num_keywords', ' global_subjectivity', ' global_sentiment_polarity', ' shares']
# run a pairplot
sns.pairplot(temp_data, vars=kw_cols, hue='popularity', diag_kind='kde')
```




    <seaborn.axisgrid.PairGrid at 0x1a569f3b90>




![png](/images/output_56_1.png)


Data Channel Evaluation
Here, it can be seen that the best articles with highest share popularity belongs to the "Others" channel. For a more concrete channel, The "Business" and "Entertaiment" channel are great for the best popularity. Coming in third position will be the "World" and "Tech". Tech channels performed generally okay. One important observation is also that "Entertaiment" channel based articls seems to be persistent in all popularity types. Meaning they might not always be the best channel to publish for.


```python
data_channel_analysis=data.groupby('data_channel').size().reset_index()
data_channel_analysis.columns=['data_channel','Number_of_articles']
data_channel_analysis.sort_values(by='Number_of_articles',ascending=False)
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
      <th>data_channel</th>
      <th>Number_of_articles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6</td>
      <td>World</td>
      <td>8168</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Tech</td>
      <td>7325</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Enter</td>
      <td>6856</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Busi</td>
      <td>6235</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Others</td>
      <td>5491</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Somed</td>
      <td>2311</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Life</td>
      <td>2077</td>
    </tr>
  </tbody>
</table>
</div>




```python

sns.catplot(x='data_channel', data=data, kind="count", height=6, aspect=.7)

```




    <seaborn.axisgrid.FacetGrid at 0x1a65fa2a90>




![png](/images/output_59_1.png)



```python
temp_data = data[(data['popularity'] == 'Very Poor') | (data['popularity'] == 'Poor') | (data['popularity'] == 'Average') | (data['popularity'] == 'Good')]
```


```python
temp_data.columns
```




    Index([' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
           ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs',
           ' num_self_hrefs', ' num_imgs', ' num_videos', 'weekdays',
           'data_channel', ' average_token_length', ' num_keywords', ' kw_min_min',
           ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max',
           ' kw_avg_max', ' kw_min_avg', ' kw_max_avg', ' kw_avg_avg',
           ' self_reference_min_shares', ' self_reference_max_shares',
           ' self_reference_avg_sharess', ' is_weekend', ' LDA_00', ' LDA_01',
           ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity',
           ' global_sentiment_polarity', ' global_rate_positive_words',
           ' global_rate_negative_words', ' rate_positive_words',
           ' rate_negative_words', ' avg_positive_polarity',
           ' min_positive_polarity', ' max_positive_polarity',
           ' avg_negative_polarity', ' min_negative_polarity',
           ' max_negative_polarity', ' title_subjectivity',
           ' title_sentiment_polarity', ' abs_title_subjectivity',
           ' abs_title_sentiment_polarity', ' shares', 'popularity'],
          dtype='object')




```python
temp_data=temp_data[['weekdays','data_channel',' shares','popularity']]
```


```python

sns.set_style("whitegrid")
ax = sns.violinplot(x="data_channel", y=" shares",data=temp_data,split=True)
ax.set_title('Distribution of shares among channels', fontsize=16);
#ax = sns.catplot(x='data_channel', col="popularity", data=temp_data, kind="count", height=10, aspect=.7)
```


![png](/images/output_63_0.png)



```python
sns.swarmplot(x="data_channel", y=" shares",data=temp_data, color='white');
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-111-a5e4958a81c0> in <module>
    ----> 1 sns.swarmplot(x="data_channel", y=" shares",data=temp_data, color='white');
    

    /opt/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py in swarmplot(x, y, hue, data, order, hue_order, dodge, orient, color, palette, size, edgecolor, linewidth, ax, **kwargs)
       2989                        linewidth=linewidth))
       2990 
    -> 2991     plotter.plot(ax, kwargs)
       2992     return ax
       2993 


    /opt/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py in plot(self, ax, kws)
       1444     def plot(self, ax, kws):
       1445         """Make the full plot."""
    -> 1446         self.draw_swarmplot(ax, kws)
       1447         self.add_legend_data(ax)
       1448         self.annotate_axes(ax)


    /opt/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py in draw_swarmplot(self, ax, kws)
       1440         for center, swarm in zip(centers, swarms):
       1441             if swarm.get_offsets().size:
    -> 1442                 self.swarm_points(ax, swarm, center, width, s, **kws)
       1443 
       1444     def plot(self, ax, kws):


    /opt/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py in swarm_points(self, ax, points, center, width, s, **kws)
       1349 
       1350         # Do the beeswarm in point coordinates
    -> 1351         new_xy = self.beeswarm(orig_xy, d)
       1352 
       1353         # Transform the point coordinates back to data coordinates


    /opt/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py in beeswarm(self, orig_xy, d)
       1311             # Find the first candidate that does not overlap any neighbours
       1312             new_xy_i = self.first_non_overlapping_candidate(candidates,
    -> 1313                                                             neighbors, d)
       1314 
       1315             # Place it into the swarm


    /opt/anaconda3/lib/python3.7/site-packages/seaborn/categorical.py in first_non_overlapping_candidate(self, candidates, neighbors, d)
       1269             dy = neighbors_y - y_i
       1270 
    -> 1271             sq_distances = np.power(dx, 2.0) + np.power(dy, 2.0)
       1272 
       1273             # good candidate does not overlap any of neighbors


    KeyboardInterrupt: 



![png](/images/output_64_1.png)



```python
fig,ax = plt.subplots(figsize=(10,10))
temp_data = data[data[' num_imgs'] <= 25]
sns.boxplot(x='popularity',y=' num_imgs', hue='data_channel', data=temp_data, showfliers=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a65b6f490>




![png](/images/output_65_1.png)


# Number of tokens in the content


```python
#n_tokens_content
sns.scatterplot(x=' n_tokens_content',y='popularity', data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a6cbaf9d0>




![png](/images/output_67_1.png)


# Title versus shares


```python
#n_tokens_title
temp_data = data[data[' shares'] <= 200000]
sns.scatterplot(x=' n_tokens_title',y=' shares', hue='popularity', data=temp_data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a6cdeb690>




![png](/images/output_69_1.png)



```python

temp_data = data[data[' shares'] <= 200000]
plt.figure(figsize=(10,10))
sns.scatterplot(x=' n_unique_tokens',y=' shares', hue='popularity', data=data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a6b61fb90>




![png](/images/output_70_1.png)



```python
#num_hrefs
temp_data = data[data[' shares'] <= 100000]
sns.scatterplot(x=' num_hrefs',y=' shares', hue='popularity', data=temp_data)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a6b767ed0>




![png](/images/output_71_1.png)


# Number of keywords in the article


```python
#num_keywords
temp_data = data[data[' shares'] <= 100000]
noise_data  = data[data[' num_keywords'] == 0]
print (noise_data.shape)
#plt.figure(figsize=(30,10))
sns.scatterplot(x=' num_keywords',y=' shares', hue='popularity', data=temp_data)
```

    (0, 49)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a6b611e10>




![png](/images/output_73_2.png)


# Length of the tokens in the article


```python
#average_token_length
temp_data = data[data[' shares'] <= 100000]
noise_data  = data[data[' average_token_length'] == 0]
print (noise_data.shape)
#plt.figure(figsize=(30,10))
sns.scatterplot(x=' average_token_length',y=' shares', hue='popularity', data=temp_data)
```

    (0, 49)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a6c9be110>




![png](/images/output_75_2.png)


# Number of videos in the content


```python
#num_videos
temp_data = data[data[' shares'] <= 100000]
noise_data  = data[data[' num_videos'] == 0]
print (noise_data.shape)
#plt.figure(figsize=(30,10))
#sns.barplot(x=' num_imgs',y=' shares', hue='popularity', data=temp_data)
sns.scatterplot(x=' num_videos', y=' shares', hue='popularity', data=temp_data)
```

    (24661, 49)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a6b664350>




![png](/images/output_77_2.png)


***Making Recommendations For Good Articles***

    n_tokens_content should be less than 1500 words. The lesser the better.
    
    n_tokens_title should be between 6 - 17 words.
    
    n_unique_tokens should be between 0.3 - 0.8
    
    num_hrefs is between 1 and 60 referrence links
    
    num_imgs should between 1 - 6 images
    
    num_videos should be between 0 - 30 vidoes. The higher the lower the odds.
    
    average_token_length should be between 4 - 10

    The number of keywords should be upto 10





### Data Transformation - Log Transform

The given data doesn't have a normal distribution. A log transformation will be carried out to transform the full data to have a normal distribution as close as possible

Normal distribution analysis for Shares
Evaluating the effects of normal distribution on the shares


```python
print("Skewness: %f" % data[' shares'].skew())
print("Kurtosis: %f" % data[' shares'].kurt())
```

    Skewness: 34.952836
    Kurtosis: 1909.975212



```python
from scipy.stats import norm,probplot
```


```python
#histogram and normal probability plot
temp_data = data[data[' shares'] <= 100000]
fig,ax = plt.subplots(figsize=(10,10))
sns.distplot(data[' shares'], fit=norm);
fig = plt.figure()
res = probplot(data[' shares'], plot=plt)
```


![png](/images/output_84_0.png)



![png](/images/output_84_1.png)



```python
#applying log transformation
new_shares_data = data.copy()
```


```python
new_shares_data.loc[new_shares_data[' shares'] > 0, ' shares'] = np.log(data.loc[data[' shares'] > 0, ' shares'])
```


```python
new_shares_log = new_shares_data[' shares']

```


```python
#transformed histogram and normal probability plot
fig,ax = plt.subplots(figsize=(10,10))
sns.distplot(new_shares_log, fit=norm);
fig = plt.figure()
res = probplot(new_shares_log, plot=plt)
```


![png](/images/output_88_0.png)



![png](/images/output_88_1.png)


Finding the normal distrubution of the dataset
Transforming the whole position data to a normal distribution

# Evaluating the impact of log transformation


```python
sns.distplot(new_shares_data[' n_tokens_content'], fit=norm)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a6e369290>




![png](/images/output_91_1.png)



```python
sns.distplot(data[' n_tokens_content'], fit=norm)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a6ef4d250>




![png](/images/output_92_1.png)


### Scalling The Data
A scaler that is immune to outliers needs to be used because there is a lot of outliers in the given data.


```python
data=data[[' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
       ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs',
       ' num_self_hrefs', ' num_imgs', ' num_videos', ' average_token_length', ' num_keywords', ' kw_min_min',
       ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max',
       ' kw_avg_max', ' kw_min_avg', ' kw_max_avg', ' kw_avg_avg',
       ' self_reference_min_shares', ' self_reference_max_shares',
       ' self_reference_avg_sharess', ' is_weekend', ' LDA_00', ' LDA_01',
       ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity',
       ' global_sentiment_polarity', ' global_rate_positive_words',
       ' global_rate_negative_words', ' rate_positive_words',
       ' rate_negative_words', ' avg_positive_polarity',
       ' min_positive_polarity', ' max_positive_polarity',
       ' avg_negative_polarity', ' min_negative_polarity',
       ' max_negative_polarity', ' title_subjectivity',
       ' title_sentiment_polarity', ' abs_title_subjectivity',
       ' abs_title_sentiment_polarity', ' shares', 'popularity']]
```


```python


# Scale features using statistics that are robust to outliers.
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

# scalled all the feature selections aside shares and populairty
scalled_data = scaler.fit_transform(data.iloc[:, :-2])

# update the dataframe back with the scalled data
data.iloc[:, :-2] = scalled_data
```

    /opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:494: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s



```python
# the data after log transformation and robust scaler
data.describe()
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
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>3.846300e+04</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
      <td>38463.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.055374</td>
      <td>0.019763</td>
      <td>-0.044655</td>
      <td>4.494114e+04</td>
      <td>-0.066021</td>
      <td>0.357541</td>
      <td>0.131590</td>
      <td>1.187687</td>
      <td>1.263786</td>
      <td>0.037661</td>
      <td>0.071671</td>
      <td>5.541637</td>
      <td>0.886038</td>
      <td>0.353131</td>
      <td>1.530201</td>
      <td>-92982.494865</td>
      <td>0.084427</td>
      <td>0.045790</td>
      <td>0.535539</td>
      <td>0.214650</td>
      <td>1.462963</td>
      <td>1.092338</td>
      <td>1.023021</td>
      <td>0.130671</td>
      <td>0.681923</td>
      <td>0.861994</td>
      <td>0.578134</td>
      <td>0.553514</td>
      <td>0.488823</td>
      <td>0.003877</td>
      <td>0.003943</td>
      <td>0.058205</td>
      <td>0.122630</td>
      <td>-0.059810</td>
      <td>0.059115</td>
      <td>0.027786</td>
      <td>-0.032476</td>
      <td>-0.050093</td>
      <td>-0.070626</td>
      <td>-0.094503</td>
      <td>-0.144014</td>
      <td>0.311147</td>
      <td>0.520642</td>
      <td>-0.472706</td>
      <td>0.619718</td>
      <td>3355.360398</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.734370</td>
      <td>0.703282</td>
      <td>0.802998</td>
      <td>8.813903e+06</td>
      <td>0.902793</td>
      <td>1.260064</td>
      <td>1.289924</td>
      <td>2.765122</td>
      <td>4.164896</td>
      <td>0.774593</td>
      <td>0.638820</td>
      <td>14.055643</td>
      <td>6.973864</td>
      <td>2.888296</td>
      <td>7.383179</td>
      <td>216395.888328</td>
      <td>0.847292</td>
      <td>0.554847</td>
      <td>2.526479</td>
      <td>1.105827</td>
      <td>10.028439</td>
      <td>5.944593</td>
      <td>5.844015</td>
      <td>0.337045</td>
      <td>1.170460</td>
      <td>1.753343</td>
      <td>0.920993</td>
      <td>0.917612</td>
      <td>0.754852</td>
      <td>0.819536</td>
      <td>0.831799</td>
      <td>0.763315</td>
      <td>0.896567</td>
      <td>0.802835</td>
      <td>0.801895</td>
      <td>0.842966</td>
      <td>1.407646</td>
      <td>0.531272</td>
      <td>0.877328</td>
      <td>0.696149</td>
      <td>1.265583</td>
      <td>0.647122</td>
      <td>1.938480</td>
      <td>0.565818</td>
      <td>0.902544</td>
      <td>11585.968776</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-5.594502</td>
      <td>-3.050697</td>
      <td>-6.288267</td>
      <td>-1.975796e+01</td>
      <td>-9.816826</td>
      <td>-0.888889</td>
      <td>-1.000000</td>
      <td>-0.333333</td>
      <td>0.000000</td>
      <td>-2.937560</td>
      <td>-2.000000</td>
      <td>0.000000</td>
      <td>-1.189189</td>
      <td>-1.103608</td>
      <td>-0.181818</td>
      <td>-843300.000000</td>
      <td>-1.555989</td>
      <td>-0.497231</td>
      <td>-1.786665</td>
      <td>-2.422724</td>
      <td>-0.600901</td>
      <td>-0.428571</td>
      <td>-0.547419</td>
      <td>0.000000</td>
      <td>-0.147319</td>
      <td>-0.265320</td>
      <td>-0.130553</td>
      <td>-0.127033</td>
      <td>-0.129530</td>
      <td>-4.233406</td>
      <td>-4.468982</td>
      <td>-1.862688</td>
      <td>-1.327227</td>
      <td>-3.817734</td>
      <td>-1.527094</td>
      <td>-3.569677</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-5.374142</td>
      <td>-1.244444</td>
      <td>-12.000000</td>
      <td>-0.250000</td>
      <td>-7.333333</td>
      <td>-1.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-0.366239</td>
      <td>-0.474026</td>
      <td>-0.521254</td>
      <td>-6.027363e-01</td>
      <td>-0.514063</td>
      <td>-0.333333</td>
      <td>-0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.486452</td>
      <td>-0.333333</td>
      <td>0.000000</td>
      <td>-0.387387</td>
      <td>-0.437558</td>
      <td>-0.181818</td>
      <td>0.000000</td>
      <td>-0.454944</td>
      <td>-0.496739</td>
      <td>-0.315841</td>
      <td>-0.405401</td>
      <td>-0.248873</td>
      <td>-0.257143</td>
      <td>-0.285610</td>
      <td>0.000000</td>
      <td>-0.036862</td>
      <td>-0.066305</td>
      <td>-0.037319</td>
      <td>-0.045663</td>
      <td>-0.055503</td>
      <td>-0.501710</td>
      <td>-0.503134</td>
      <td>-0.476963</td>
      <td>-0.465473</td>
      <td>-0.541872</td>
      <td>-0.458128</td>
      <td>-0.493147</td>
      <td>-1.000000</td>
      <td>-0.500000</td>
      <td>-0.534287</td>
      <td>-0.533333</td>
      <td>-0.333333</td>
      <td>-0.250000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>945.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1400.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>0.633761</td>
      <td>0.525974</td>
      <td>0.478746</td>
      <td>3.972637e-01</td>
      <td>0.485937</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.513548</td>
      <td>0.666667</td>
      <td>1.000000</td>
      <td>0.612613</td>
      <td>0.562442</td>
      <td>0.818182</td>
      <td>0.000000</td>
      <td>0.545056</td>
      <td>0.503261</td>
      <td>0.684159</td>
      <td>0.594599</td>
      <td>0.751127</td>
      <td>0.742857</td>
      <td>0.714390</td>
      <td>0.000000</td>
      <td>0.963138</td>
      <td>0.933695</td>
      <td>0.962681</td>
      <td>0.954337</td>
      <td>0.944497</td>
      <td>0.498290</td>
      <td>0.496866</td>
      <td>0.523037</td>
      <td>0.534527</td>
      <td>0.458128</td>
      <td>0.541872</td>
      <td>0.506853</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.465713</td>
      <td>0.466667</td>
      <td>0.666667</td>
      <td>0.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2700.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2.895242</td>
      <td>2.896457</td>
      <td>29.014405</td>
      <td>1.728581e+09</td>
      <td>38.126824</td>
      <td>32.888889</td>
      <td>37.666667</td>
      <td>42.333333</td>
      <td>91.000000</td>
      <td>9.209366</td>
      <td>1.000000</td>
      <td>75.600000</td>
      <td>536.468468</td>
      <td>196.994714</td>
      <td>109.337662</td>
      <td>0.000000</td>
      <td>3.864392</td>
      <td>1.281989</td>
      <td>121.870096</td>
      <td>34.602216</td>
      <td>421.682524</td>
      <td>120.042857</td>
      <td>200.164906</td>
      <td>1.000000</td>
      <td>3.938617</td>
      <td>7.102195</td>
      <td>2.871562</td>
      <td>2.815463</td>
      <td>2.272446</td>
      <td>5.038877</td>
      <td>5.239886</td>
      <td>5.450351</td>
      <td>14.332228</td>
      <td>1.527094</td>
      <td>3.817734</td>
      <td>6.294789</td>
      <td>18.000000</td>
      <td>0.500000</td>
      <td>1.866081</td>
      <td>1.244444</td>
      <td>1.333333</td>
      <td>1.750000</td>
      <td>7.333333</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>843300.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Data Clustering - Grouping Similar Articles together.
### Here, we are going to find any special pattern from the data. Using unsupervised learning to find cluster from the data and then create a new set of data out from the clusters formed. The idea is that it is easy to build a model for similar clustered articles than to build a model for all the articles.


```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
```


```python
# Kmeans perform poorly on high feature space
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.iloc[:,:-2])
reduced_data.shape
```




    (38463, 2)




```python
# plotting the clusters PCA
plt.figure(figsize=(10,10))
plt.plot(reduced_data[:,0], reduced_data[:,1], 'r.')
plt.title('PCA Transformation')

plt.show()
```


![png](/images/output_100_0.png)



```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, n_iter=300)
reduced_tsne = tsne.fit_transform(data.iloc[:,:-2])

# plotting the clusters TSNE
plt.figure(figsize=(10,10))
plt.plot(reduced_tsne[:,0], reduced_tsne[:,1], 'r.')
plt.title('TSNE Transformation')
plt.show()
```


![png](/images/output_101_0.png)


### Machine learning Model



```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
```


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```


```python
from sklearn.metrics import accuracy_score
```


```python
data.columns
```




    Index([' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
           ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs',
           ' num_self_hrefs', ' num_imgs', ' num_videos', 'weekdays',
           'data_channel', ' average_token_length', ' num_keywords', ' kw_min_min',
           ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max',
           ' kw_avg_max', ' kw_min_avg', ' kw_max_avg', ' kw_avg_avg',
           ' self_reference_min_shares', ' self_reference_max_shares',
           ' self_reference_avg_sharess', ' is_weekend', ' LDA_00', ' LDA_01',
           ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity',
           ' global_sentiment_polarity', ' global_rate_positive_words',
           ' global_rate_negative_words', ' rate_positive_words',
           ' rate_negative_words', ' avg_positive_polarity',
           ' min_positive_polarity', ' max_positive_polarity',
           ' avg_negative_polarity', ' min_negative_polarity',
           ' max_negative_polarity', ' title_subjectivity',
           ' title_sentiment_polarity', ' abs_title_subjectivity',
           ' abs_title_sentiment_polarity', ' shares', 'popularity'],
          dtype='object')




```python

share_label = list()
for share in data[' shares']:
    if share <= 1400:
        share_label.append(0)
    else:
        share_label.append(1)

# Update this class label into the dataframe
data = pd.concat([data.reset_index(drop=True), pd.DataFrame(share_label, columns=['popularity_classify'])], axis=1)
```


```python
data.head(3)
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
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>weekdays</th>
      <th>data_channel</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
      <th>popularity</th>
      <th>popularity</th>
      <th>popularity</th>
      <th>popularity_classify</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>12.0</td>
      <td>219.0</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Enter</td>
      <td>4.680365</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>0.0</td>
      <td>0.500331</td>
      <td>0.378279</td>
      <td>0.040005</td>
      <td>0.041263</td>
      <td>0.040123</td>
      <td>0.521617</td>
      <td>0.092562</td>
      <td>0.045662</td>
      <td>0.013699</td>
      <td>0.769231</td>
      <td>0.230769</td>
      <td>0.378636</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600</td>
      <td>-0.200000</td>
      <td>0.5</td>
      <td>-0.1875</td>
      <td>0.0</td>
      <td>0.1875</td>
      <td>593</td>
      <td>Very Poor</td>
      <td>Unpopular</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9.0</td>
      <td>255.0</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Busi</td>
      <td>4.913725</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.799756</td>
      <td>0.050047</td>
      <td>0.050096</td>
      <td>0.050101</td>
      <td>0.050001</td>
      <td>0.341246</td>
      <td>0.148948</td>
      <td>0.043137</td>
      <td>0.015686</td>
      <td>0.733333</td>
      <td>0.266667</td>
      <td>0.286915</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125</td>
      <td>-0.100000</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>711</td>
      <td>Poor</td>
      <td>Unpopular</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>9.0</td>
      <td>211.0</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monday</td>
      <td>Busi</td>
      <td>4.393365</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>0.0</td>
      <td>0.217792</td>
      <td>0.033334</td>
      <td>0.033351</td>
      <td>0.033334</td>
      <td>0.682188</td>
      <td>0.702222</td>
      <td>0.323333</td>
      <td>0.056872</td>
      <td>0.009479</td>
      <td>0.857143</td>
      <td>0.142857</td>
      <td>0.495833</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800</td>
      <td>-0.133333</td>
      <td>0.0</td>
      <td>0.0000</td>
      <td>0.5</td>
      <td>0.0000</td>
      <td>1500</td>
      <td>Good</td>
      <td>Popular</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data=data.drop(['popularity','popularity','popularity'],axis=1)
```


```python
data=data.drop(['weekdays','data_channel'],axis=1)
```


```python
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

```


```python
# Fitting the linear model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Instantiate & fit the DT
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)

SVM_model = LinearSVC()
SVM_model.fit(X_train, y_train)


# Instantiate the model & fit it to our data
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)

my_random_forest = RandomForestClassifier(n_estimators=50)
my_random_forest.fit(X_train, y_train)

```

    /opt/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=50,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
print(f"The TRAIN classification accuracy is:  {logreg_model.score(X_train,y_train)}")
print(f"The TEST classification accuracy is:  {logreg_model.score(X_test,y_test)}")


print(f"The TRAIN classification accuracy is:  {DT_model.score(X_train,y_train)}")
print(f"The TEST classification accuracy is:  {DT_model.score(X_test,y_test)}")

print(f"The TRAIN classification accuracy is: {SVM_model.score(X_train,y_train)}")
print(f"The TEST classification accuracy is: {SVM_model.score(X_test,y_test)}")

print(f"The TRAIN classification accuracy is: {KNN_model.score(X_train,y_train)}")
print(f"The TEST classification accuracy is: {KNN_model.score(X_test,y_test)}")

print(f"Random Forest training accuracy: {my_random_forest.score(X_train, y_train)}")
print(f"Random Forest testing accuracy: {my_random_forest.score(X_test, y_test)}")

```

    The TRAIN classification accuracy is:  0.9637498142920814
    The TEST classification accuracy is:  0.9652482884132074
    The TRAIN classification accuracy is:  1.0
    The TEST classification accuracy is:  1.0
    The TRAIN classification accuracy is: 0.6850765116624573
    The TEST classification accuracy is: 0.6887945229222636
    The TRAIN classification accuracy is: 0.823020353587877
    The TEST classification accuracy is: 0.7167865499610018
    Random Forest training accuracy: 1.0
    Random Forest testing accuracy: 1.0



```python
#Score it on logistic reression model


y_prediction_logreg = logreg_model.predict(X_test)


y_prediction_DT = DT_model.predict(X_test)


y_prediction_SVM = SVM_model.predict(X_test)

# Score the model on the KNN-test set

y_predictions_knn = KNN_model.predict(X_test)


y_predictions_RF=my_random_forest.predict(X_test)

```


```python
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("PRediction_Accuracy_logreg:",metrics.accuracy_score(y_test, y_prediction_logreg))
print("PRediction_Accuracy_DT:",metrics.accuracy_score(y_test, y_prediction_DT))
print("PRediction_Accuracy_SVM:",metrics.accuracy_score(y_test, y_prediction_SVM))
print("PRediction_Accuracy_KNN:",metrics.accuracy_score(y_test, y_predictions_knn))
print("PRediction_Accuracy_RF:",metrics.accuracy_score(y_test, y_predictions_RF))
```

    PRediction_Accuracy_logreg: 0.9652482884132074
    PRediction_Accuracy_DT: 1.0
    PRediction_Accuracy_SVM: 0.6101915243955283
    PRediction_Accuracy_KNN: 0.7167865499610018
    PRediction_Accuracy_RF: 1.0

