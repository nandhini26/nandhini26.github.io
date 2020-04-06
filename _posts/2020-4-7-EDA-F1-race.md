### F1 Race EDA 

### Driver Standings csv


```python
# import stand libraries

import numpy as np
import pandas as pd
```


```python
#Import driver standings file

df_driver_standings=pd.read_csv('./Capstone_core_data/driver_standings.csv')
```


```python
# droping unnecessary columns

df_driver_standings=df_driver_standings.drop(['driverStandingsId','positionText'],axis=1)
```


```python
# Selecting the necessary columns

df_driver_standings.columns=['raceId','driverId','D_points','D-position','D_wins']
```


```python
df_driver_standings.head()
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
      <th>raceId</th>
      <th>driverId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>10.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>18</td>
      <td>2</td>
      <td>8.0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>18</td>
      <td>3</td>
      <td>6.0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>18</td>
      <td>4</td>
      <td>5.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>18</td>
      <td>5</td>
      <td>4.0</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Selecting the 1st position drivers as separate list "df-podium"


```python
df_podium=df_driver_standings.loc[df_driver_standings['D-position']==1]
```


```python
df_driver=pd.read_csv('./Capstone_core_data/drivers.csv')
```


```python
df_podium.head(2)
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
      <th>raceId</th>
      <th>driverId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>10.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>19</td>
      <td>1</td>
      <td>14.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_podium=df_podium.groupby(['driverId','D-position']).raceId.count().reset_index().rename(columns={"raceId":"Position_1"})
```


```python
# arranging the list based on the total number of times a drivers securedpodium position
df_podium.head(10)
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
      <th>driverId</th>
      <th>D-position</th>
      <th>Position_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>103</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>29</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>52</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>13</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>14</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <td>7</td>
      <td>17</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <td>8</td>
      <td>18</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <td>9</td>
      <td>20</td>
      <td>1</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_podium=pd.merge(df_driver,df_podium,how='inner',on='driverId')
```


```python
df_podium_analysis=df_podium.sort_values(by='Position_1',ascending=False).head(15)
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
plt.figure(figsize=(10,5))
sns.barplot(df_podium_analysis.forename.head(10), df_podium_analysis.Position_1, alpha=1)
plt.title('Highest Podiums by the Driver')
plt.ylabel('Podiums reached by the driver', fontsize=12)
plt.xlabel('Driver Names', fontsize=16)
plt.show()
```


![png](output_16_0.png)


---
### Findings:

Mochael Schumacher has played more number of races than Lewis. But as we all know he met a tragic accident,and he is been hospitalized since then,Lewis HAmilton is the next racer who drove in highest number of races.

---
    


```python
df_podium.to_csv('./intermediate_files/position_1.csv')
```


```python
# Checking the size of the driver standings file
df_driver_standings.shape
```




    (32566, 5)




```python
df_driver.head()
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
      <th>driverId</th>
      <th>driverRef</th>
      <th>number</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
      <th>dob</th>
      <th>nationality</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>hamilton</td>
      <td>44</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
      <td>1985-01-07</td>
      <td>British</td>
      <td>http://en.wikipedia.org/wiki/Lewis_Hamilton</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>heidfeld</td>
      <td>\N</td>
      <td>HEI</td>
      <td>Nick</td>
      <td>Heidfeld</td>
      <td>1977-05-10</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nick_Heidfeld</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>rosberg</td>
      <td>6</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
      <td>1985-06-27</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nico_Rosberg</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>alonso</td>
      <td>14</td>
      <td>ALO</td>
      <td>Fernando</td>
      <td>Alonso</td>
      <td>1981-07-29</td>
      <td>Spanish</td>
      <td>http://en.wikipedia.org/wiki/Fernando_Alonso</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>kovalainen</td>
      <td>\N</td>
      <td>KOV</td>
      <td>Heikki</td>
      <td>Kovalainen</td>
      <td>1981-10-19</td>
      <td>Finnish</td>
      <td>http://en.wikipedia.org/wiki/Heikki_Kovalainen</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Picking driverID,driver code and firstName of the driver for further analysis

df_driver=df_driver[['driverId','code','forename','surname']]
```


```python
#df race played is the data frame from driver standings 
#grouped by raceID count (total number of races drove by each of the driver) for further analysis

df_race_played=df_driver_standings.groupby(['driverId']).raceId.count().reset_index().rename(columns={"raceId":"TotalRacesplayed"})

```


```python
# How many number of races played by each drivers

df_race_played
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
      <th>driverId</th>
      <th>TotalRacesplayed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>249</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>193</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>206</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>316</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>111</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>835</td>
      <td>844</td>
      <td>42</td>
    </tr>
    <tr>
      <td>836</td>
      <td>845</td>
      <td>21</td>
    </tr>
    <tr>
      <td>837</td>
      <td>846</td>
      <td>21</td>
    </tr>
    <tr>
      <td>838</td>
      <td>847</td>
      <td>21</td>
    </tr>
    <tr>
      <td>839</td>
      <td>848</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>840 rows × 2 columns</p>
</div>




```python
# df_driver_name_races_played is the data frame merged from race played with driver to get the driver name 
# for durther analysis

df_driver_name_races_played=pd.merge(df_race_played,df_driver,how='inner',on='driverId')
```


```python
df_driver_name_races_played.columns
```




    Index(['driverId', 'TotalRacesplayed', 'code', 'forename', 'surname'], dtype='object')




```python
df_driver_name_races_played=df_driver_name_races_played[['TotalRacesplayed', 'forename','surname','driverId']]
```

### Sorting the df_driver_name_races_played based on the  "Total race played" 


```python

df_driver_name_races_played=df_driver_name_races_played.sort_values(by='TotalRacesplayed',ascending=False)

```


```python
# for easy understanding,driver names brought in

df_driver_name_races_played.head(20)
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
      <th>TotalRacesplayed</th>
      <th>forename</th>
      <th>surname</th>
      <th>driverId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>17</td>
      <td>316</td>
      <td>Jenson</td>
      <td>Button</td>
      <td>18</td>
    </tr>
    <tr>
      <td>3</td>
      <td>316</td>
      <td>Fernando</td>
      <td>Alonso</td>
      <td>4</td>
    </tr>
    <tr>
      <td>7</td>
      <td>314</td>
      <td>Kimi</td>
      <td>Raikkonen</td>
      <td>8</td>
    </tr>
    <tr>
      <td>28</td>
      <td>312</td>
      <td>Michael</td>
      <td>Schumacher</td>
      <td>30</td>
    </tr>
    <tr>
      <td>21</td>
      <td>308</td>
      <td>Rubens</td>
      <td>Barrichello</td>
      <td>22</td>
    </tr>
    <tr>
      <td>12</td>
      <td>275</td>
      <td>Felipe</td>
      <td>Massa</td>
      <td>13</td>
    </tr>
    <tr>
      <td>115</td>
      <td>260</td>
      <td>Riccardo</td>
      <td>Patrese</td>
      <td>119</td>
    </tr>
    <tr>
      <td>14</td>
      <td>250</td>
      <td>Jarno</td>
      <td>Trulli</td>
      <td>15</td>
    </tr>
    <tr>
      <td>0</td>
      <td>249</td>
      <td>Lewis</td>
      <td>Hamilton</td>
      <td>1</td>
    </tr>
    <tr>
      <td>19</td>
      <td>240</td>
      <td>Sebastian</td>
      <td>Vettel</td>
      <td>20</td>
    </tr>
    <tr>
      <td>13</td>
      <td>239</td>
      <td>David</td>
      <td>Coulthard</td>
      <td>14</td>
    </tr>
    <tr>
      <td>20</td>
      <td>229</td>
      <td>Giancarlo</td>
      <td>Fisichella</td>
      <td>21</td>
    </tr>
    <tr>
      <td>91</td>
      <td>220</td>
      <td>Nigel</td>
      <td>Mansell</td>
      <td>95</td>
    </tr>
    <tr>
      <td>101</td>
      <td>218</td>
      <td>Michele</td>
      <td>Alboreto</td>
      <td>105</td>
    </tr>
    <tr>
      <td>106</td>
      <td>216</td>
      <td>Andrea</td>
      <td>de Cesaris</td>
      <td>110</td>
    </tr>
    <tr>
      <td>74</td>
      <td>215</td>
      <td>Gerhard</td>
      <td>Berger</td>
      <td>77</td>
    </tr>
    <tr>
      <td>16</td>
      <td>212</td>
      <td>Mark</td>
      <td>Webber</td>
      <td>17</td>
    </tr>
    <tr>
      <td>129</td>
      <td>209</td>
      <td>Nelson</td>
      <td>Piquet</td>
      <td>137</td>
    </tr>
    <tr>
      <td>2</td>
      <td>206</td>
      <td>Nico</td>
      <td>Rosberg</td>
      <td>3</td>
    </tr>
    <tr>
      <td>113</td>
      <td>204</td>
      <td>Alain</td>
      <td>Prost</td>
      <td>117</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Get the top 10 drivers based on the highest number of races played 

df=df_driver_name_races_played
```

### Remember this par of the analysis does not consider whether they have raced till 2019


```python
df.to_csv('./intermediate_files/Highest_grand_prix_entered.csv')
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
plt.figure(figsize=(8,5))
sns.barplot(df_driver_name_races_played.forename.head(10), df_driver_name_races_played.TotalRacesplayed, alpha=1)
plt.title('Races played by Drivers')
plt.ylabel('Number of races played', fontsize=12)
plt.xlabel('Driver Names', fontsize=16)
plt.show()
```


![png](output_34_0.png)


### ***Findings*** ###
    

---
Even though ***Lewis Hamilton*** played very less number of races, he has won the world championship title for consequently ***6*** years.Again,this part of analysis doesn't consider how active the racers are in 2019 races.

---

### Races.csv


```python
# race details file
df_race=pd.read_csv('./Capstone_core_data/races.csv')
```


```python
df_race.head()
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
      <th>raceId</th>
      <th>year</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
      <th>date</th>
      <th>time</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2009</td>
      <td>1</td>
      <td>1</td>
      <td>Australian Grand Prix</td>
      <td>2009-03-29</td>
      <td>06:00:00</td>
      <td>http://en.wikipedia.org/wiki/2009_Australian_G...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2009</td>
      <td>2</td>
      <td>2</td>
      <td>Malaysian Grand Prix</td>
      <td>2009-04-05</td>
      <td>09:00:00</td>
      <td>http://en.wikipedia.org/wiki/2009_Malaysian_Gr...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2009</td>
      <td>3</td>
      <td>17</td>
      <td>Chinese Grand Prix</td>
      <td>2009-04-19</td>
      <td>07:00:00</td>
      <td>http://en.wikipedia.org/wiki/2009_Chinese_Gran...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2009</td>
      <td>4</td>
      <td>3</td>
      <td>Bahrain Grand Prix</td>
      <td>2009-04-26</td>
      <td>12:00:00</td>
      <td>http://en.wikipedia.org/wiki/2009_Bahrain_Gran...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2009</td>
      <td>5</td>
      <td>4</td>
      <td>Spanish Grand Prix</td>
      <td>2009-05-10</td>
      <td>12:00:00</td>
      <td>http://en.wikipedia.org/wiki/2009_Spanish_Gran...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_race=df_race[['raceId','year','round','circuitId','name']]
```


```python
df_race.head()
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
      <th>raceId</th>
      <th>year</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2009</td>
      <td>1</td>
      <td>1</td>
      <td>Australian Grand Prix</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2009</td>
      <td>2</td>
      <td>2</td>
      <td>Malaysian Grand Prix</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2009</td>
      <td>3</td>
      <td>17</td>
      <td>Chinese Grand Prix</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2009</td>
      <td>4</td>
      <td>3</td>
      <td>Bahrain Grand Prix</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2009</td>
      <td>5</td>
      <td>4</td>
      <td>Spanish Grand Prix</td>
    </tr>
  </tbody>
</table>
</div>




```python
# counting each race counts

df_race['name'].value_counts().head(10)
```




    British Grand Prix          71
    Italian Grand Prix          71
    Monaco Grand Prix           67
    Belgian Grand Prix          65
    German Grand Prix           64
    French Grand Prix           61
    Canadian Grand Prix         51
    Spanish Grand Prix          50
    Brazilian Grand Prix        48
    United States Grand Prix    42
    Name: name, dtype: int64




```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
race_count  = df_race['name'].value_counts()
race_count = race_count[:10,]
plt.figure(figsize=(10,5))
sns.barplot(race_count.index, race_count.values, alpha=1)
plt.xticks(fontsize=16,rotation=75)
plt.title('Races in a Season',fontsize=22)
plt.ylabel('Number of Occurrences', fontsize=16)

plt.show()
```


![png](output_44_0.png)


### ***Findings*** ###

---

Over the years, British grand prix hosts higest number of F1 races.

---


```python
df_race.shape

```




    (1040, 5)




```python
df_driver_standings.shape
```




    (32566, 5)




```python

df_driver_all_races=pd.merge(df_driver_standings,df_race,how='left',on='raceId')
df_driver_all_races.shape
```




    (32566, 9)




```python
driver_standings_names=pd.merge(df_driver_all_races,df_driver,how='inner',on='driverId')
```


```python
driver_standings_names.columns
```




    Index(['raceId', 'driverId', 'D_points', 'D-position', 'D_wins', 'year',
           'round', 'circuitId', 'name', 'code', 'forename', 'surname'],
          dtype='object')



### Calculating driver points across years ###


```python
driver_standings_names.to_csv('./intermediate_files/driverstandingswith_names.csv')
```



### Analysis purpose we can calculate the points of a driver across year across races


```python
df_driver_max_points=driver_standings_names.groupby(['driverId','year']).max()
```


```python
df_driver_max_points=df_driver_max_points.sort_values(by='D_points',ascending=False)
```


```python
df_driver_max_points.reset_index()
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
      <th>driverId</th>
      <th>year</th>
      <th>raceId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2019</td>
      <td>1030</td>
      <td>413.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>2018</td>
      <td>1009</td>
      <td>408.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2</td>
      <td>20</td>
      <td>2013</td>
      <td>899</td>
      <td>397.0</td>
      <td>3</td>
      <td>13</td>
      <td>19</td>
      <td>69</td>
      <td>United States Grand Prix</td>
      <td>VET</td>
      <td>Sebastian</td>
      <td>Vettel</td>
    </tr>
    <tr>
      <td>3</td>
      <td>20</td>
      <td>2011</td>
      <td>859</td>
      <td>392.0</td>
      <td>1</td>
      <td>11</td>
      <td>19</td>
      <td>68</td>
      <td>Turkish Grand Prix</td>
      <td>VET</td>
      <td>Sebastian</td>
      <td>Vettel</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3</td>
      <td>2016</td>
      <td>968</td>
      <td>385.0</td>
      <td>2</td>
      <td>9</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
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
    </tr>
    <tr>
      <td>3074</td>
      <td>411</td>
      <td>1962</td>
      <td>737</td>
      <td>0.0</td>
      <td>37</td>
      <td>0</td>
      <td>9</td>
      <td>56</td>
      <td>South African Grand Prix</td>
      <td>\N</td>
      <td>Doug</td>
      <td>Serrurier</td>
    </tr>
    <tr>
      <td>3075</td>
      <td>410</td>
      <td>1965</td>
      <td>708</td>
      <td>0.0</td>
      <td>42</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>United States Grand Prix</td>
      <td>\N</td>
      <td>Neville</td>
      <td>Lederle</td>
    </tr>
    <tr>
      <td>3076</td>
      <td>410</td>
      <td>1963</td>
      <td>728</td>
      <td>0.0</td>
      <td>51</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>South African Grand Prix</td>
      <td>\N</td>
      <td>Neville</td>
      <td>Lederle</td>
    </tr>
    <tr>
      <td>3077</td>
      <td>409</td>
      <td>1965</td>
      <td>708</td>
      <td>0.0</td>
      <td>40</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>United States Grand Prix</td>
      <td>\N</td>
      <td>Trevor</td>
      <td>Blokdyk</td>
    </tr>
    <tr>
      <td>3078</td>
      <td>11</td>
      <td>2008</td>
      <td>35</td>
      <td>0.0</td>
      <td>21</td>
      <td>0</td>
      <td>18</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>SAT</td>
      <td>Takuma</td>
      <td>Sato</td>
    </tr>
  </tbody>
</table>
<p>3079 rows × 12 columns</p>
</div>




```python
df_driver_max_points.columns
```




    Index(['raceId', 'D_points', 'D-position', 'D_wins', 'round', 'circuitId',
           'name', 'code', 'forename', 'surname'],
          dtype='object')




```python
#df_driver_max_points.loc[(df_driver_max_points['driverId']==1)]

df_driver_max_points_analysis=df_driver_max_points.loc[(df_driver_max_points['forename']=='Lewis')]
```


```python
df_driver_max_points_analysis
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
      <th></th>
      <th>raceId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
    </tr>
    <tr>
      <th>driverId</th>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="13" valign="top">1</td>
      <td>2019</td>
      <td>1030</td>
      <td>413.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2018</td>
      <td>1009</td>
      <td>408.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2014</td>
      <td>918</td>
      <td>384.0</td>
      <td>19</td>
      <td>11</td>
      <td>19</td>
      <td>71</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2015</td>
      <td>945</td>
      <td>381.0</td>
      <td>1</td>
      <td>10</td>
      <td>19</td>
      <td>71</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2016</td>
      <td>968</td>
      <td>380.0</td>
      <td>3</td>
      <td>10</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2017</td>
      <td>988</td>
      <td>363.0</td>
      <td>2</td>
      <td>9</td>
      <td>20</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2010</td>
      <td>355</td>
      <td>240.0</td>
      <td>7</td>
      <td>3</td>
      <td>19</td>
      <td>35</td>
      <td>Turkish Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2011</td>
      <td>859</td>
      <td>227.0</td>
      <td>5</td>
      <td>3</td>
      <td>19</td>
      <td>68</td>
      <td>Turkish Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2012</td>
      <td>879</td>
      <td>190.0</td>
      <td>5</td>
      <td>4</td>
      <td>20</td>
      <td>69</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2013</td>
      <td>899</td>
      <td>189.0</td>
      <td>5</td>
      <td>1</td>
      <td>19</td>
      <td>69</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2007</td>
      <td>52</td>
      <td>109.0</td>
      <td>3</td>
      <td>4</td>
      <td>17</td>
      <td>20</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2008</td>
      <td>35</td>
      <td>98.0</td>
      <td>4</td>
      <td>5</td>
      <td>18</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2009</td>
      <td>17</td>
      <td>49.0</td>
      <td>11</td>
      <td>2</td>
      <td>17</td>
      <td>24</td>
      <td>Turkish Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_driver_max_points_analysis=df_driver_max_points_analysis.groupby('name').D_points.sum().reset_index().rename(columns={"D_points":"Total_points"})
```


```python
df_driver_max_points_analysis
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
      <th>name</th>
      <th>Total_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Turkish Grand Prix</td>
      <td>614.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>United States Grand Prix</td>
      <td>2817.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
plt.figure(figsize=(8,5))
sns.barplot(df_driver_max_points_analysis.name, df_driver_max_points_analysis.Total_points, alpha=1)
plt.xlabel('Driver Names', fontsize=16)
plt.title('Points scored by races',fontsize=22)
plt.ylabel('Max points', fontsize=16)
plt.xticks(fontsize=16,rotation=85)
plt.show()
```


![png](output_64_0.png)


---
### Findings:

    Lewis Hamilton scores more points in US grand prix than Turkish Grand prix.
    
    
----


```python
df_driver_max_points_analysis2=df_driver_max_points.groupby('name').D_points.sum().reset_index().rename(columns={"D_points":"Total_points"})
```


```python
df_driver_max_points_analysis2=df_driver_max_points_analysis2.sort_values(by='Total_points',ascending=False).head(5)
```


```python
df_driver_max_points_analysis2
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
      <th>name</th>
      <th>Total_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20</td>
      <td>United States Grand Prix</td>
      <td>26787.00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Turkish Grand Prix</td>
      <td>5197.50</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Spanish Grand Prix</td>
      <td>4526.00</td>
    </tr>
    <tr>
      <td>21</td>
      <td>United States Grand Prix West</td>
      <td>3014.00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Swiss Grand Prix</td>
      <td>784.42</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8,5))
sns.barplot(df_driver_max_points_analysis2.name.head(5), df_driver_max_points_analysis2.Total_points.head(5), alpha=1)
plt.xlabel('Driver Names', fontsize=16)
plt.title('Points scored by races',fontsize=22)
plt.ylabel('Max points', fontsize=16)
plt.xticks(fontsize=16,rotation=85)
plt.show()
```


![png](output_69_0.png)


---

### Findings:
    
    Overall, racers scored high points in US grand prix races than other races.
    
    
----


```python
df_driver_max_points
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
      <th></th>
      <th>raceId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
    </tr>
    <tr>
      <th>driverId</th>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" valign="top">1</td>
      <td>2019</td>
      <td>1030</td>
      <td>413.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2018</td>
      <td>1009</td>
      <td>408.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">20</td>
      <td>2013</td>
      <td>899</td>
      <td>397.0</td>
      <td>3</td>
      <td>13</td>
      <td>19</td>
      <td>69</td>
      <td>United States Grand Prix</td>
      <td>VET</td>
      <td>Sebastian</td>
      <td>Vettel</td>
    </tr>
    <tr>
      <td>2011</td>
      <td>859</td>
      <td>392.0</td>
      <td>1</td>
      <td>11</td>
      <td>19</td>
      <td>68</td>
      <td>Turkish Grand Prix</td>
      <td>VET</td>
      <td>Sebastian</td>
      <td>Vettel</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016</td>
      <td>968</td>
      <td>385.0</td>
      <td>2</td>
      <td>9</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
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
    </tr>
    <tr>
      <td>411</td>
      <td>1962</td>
      <td>737</td>
      <td>0.0</td>
      <td>37</td>
      <td>0</td>
      <td>9</td>
      <td>56</td>
      <td>South African Grand Prix</td>
      <td>\N</td>
      <td>Doug</td>
      <td>Serrurier</td>
    </tr>
    <tr>
      <td rowspan="2" valign="top">410</td>
      <td>1965</td>
      <td>708</td>
      <td>0.0</td>
      <td>42</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>United States Grand Prix</td>
      <td>\N</td>
      <td>Neville</td>
      <td>Lederle</td>
    </tr>
    <tr>
      <td>1963</td>
      <td>728</td>
      <td>0.0</td>
      <td>51</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>South African Grand Prix</td>
      <td>\N</td>
      <td>Neville</td>
      <td>Lederle</td>
    </tr>
    <tr>
      <td>409</td>
      <td>1965</td>
      <td>708</td>
      <td>0.0</td>
      <td>40</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>United States Grand Prix</td>
      <td>\N</td>
      <td>Trevor</td>
      <td>Blokdyk</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2008</td>
      <td>35</td>
      <td>0.0</td>
      <td>21</td>
      <td>0</td>
      <td>18</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>SAT</td>
      <td>Takuma</td>
      <td>Sato</td>
    </tr>
  </tbody>
</table>
<p>3079 rows × 10 columns</p>
</div>




```python
df_win=df_driver_max_points.groupby(['driverId']).D_wins.sum().reset_index().rename(columns={"D_wins":"LifeTimeWinning"})

```


```python
df_life_time_winning=df_win.sort_values(by='LifeTimeWinning',ascending=False)
```


```python
df_life_time_winning.columns
```




    Index(['driverId', 'LifeTimeWinning'], dtype='object')




```python
df_merge2=pd.merge(df_life_time_winning,df_driver,how='inner',on='driverId')
```


```python
df_merge2.head()
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
      <th>driverId</th>
      <th>LifeTimeWinning</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>30</td>
      <td>91</td>
      <td>MSC</td>
      <td>Michael</td>
      <td>Schumacher</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>84</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2</td>
      <td>20</td>
      <td>53</td>
      <td>VET</td>
      <td>Sebastian</td>
      <td>Vettel</td>
    </tr>
    <tr>
      <td>3</td>
      <td>117</td>
      <td>51</td>
      <td>\N</td>
      <td>Alain</td>
      <td>Prost</td>
    </tr>
    <tr>
      <td>4</td>
      <td>102</td>
      <td>41</td>
      <td>\N</td>
      <td>Ayrton</td>
      <td>Senna</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merge2.to_csv('./intermediate_files/Lifetime_wins.csv')
```


```python
plt.figure(figsize=(10,5))
sns.barplot(df_merge2.forename.head(10),df_merge2.LifeTimeWinning.head(10), alpha=1)
plt.title('Total number of races won by Drivers',fontsize=18)
plt.ylabel('Number of times winning across the grand prix', fontsize=12)
plt.xlabel('Driver Names', fontsize=16)
plt.show()
```


![png](output_78_0.png)


### ***Findings*** ###
***Michael schumacher*** is the leading driver who has played more number of races in grand prix than Lewis.As we all know Michael  schumacher met tragic accident in ice skiiing ,currently,***Lewis*** is the leading driver. 


***Life time points Calculation***


```python
df_tr=df_driver_max_points.groupby(['driverId','year']).sum()
```


```python
df_tr.columns
```




    Index(['raceId', 'D_points', 'D-position', 'D_wins', 'round', 'circuitId'], dtype='object')




```python
df_tr=df_tr.drop(['D-position'],axis=1)
```


```python
df_tr=df_tr.groupby(['driverId']).D_points.sum().reset_index().rename(columns={"D_points":"Lifetimepoints"})

```


```python
df_tr.columns
```




    Index(['driverId', 'Lifetimepoints'], dtype='object')




```python
df_tr.sort_values(by='Lifetimepoints',ascending=False)
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
      <th>driverId</th>
      <th>Lifetimepoints</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>3431.0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>20</td>
      <td>2985.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1899.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>1859.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1594.5</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>407</td>
      <td>415</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>408</td>
      <td>416</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>409</td>
      <td>417</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>411</td>
      <td>419</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>420</td>
      <td>428</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>840 rows × 2 columns</p>
</div>




```python
df_life_time_points=pd.merge(df_tr,df_driver,how='inner',on='driverId')
```


```python
df_life_time_points.columns
```




    Index(['driverId', 'Lifetimepoints', 'code', 'forename', 'surname'], dtype='object')




```python
df_life_time_points=df_life_time_points[['Lifetimepoints', 'forename','surname','driverId']]
```


```python
df_life_time_points.columns
```




    Index(['Lifetimepoints', 'forename', 'surname', 'driverId'], dtype='object')




```python
df_life_time_points=df_life_time_points.sort_values(by='Lifetimepoints',ascending=False)
```


```python
df_life_time_points.to_csv('life_time_points.csv')
```


```python
plt.figure(figsize=(10,5))
sns.barplot(df_life_time_points.forename.head(10),df_life_time_points.Lifetimepoints.head(10), alpha=1)
plt.title('Lifetime points of the driver',fontsize=18)
plt.ylabel('Points scored', fontsize=18)
plt.xlabel('Driver first name', fontsize=18)
plt.show()
```


![png](output_93_0.png)


### ***Findings*** ####

---

***Lewis Hamilton*** scored the higest number of points(***3431***) across all the races across all the years.

---

### Df_driver points is the summary of driver cumulative points in that year across all 20 races


```python
df_driver_max_points.to_csv("./intermediate_files/df_driver_max_points.csv")
```


```python
df_driver_max_points=pd.read_csv("./intermediate_files/df_driver_max_points.csv",index_col=0)
```


```python
df_driver_max_points=df_driver_max_points
```


```python
df_driver_max_points.columns
```




    Index(['year', 'raceId', 'D_points', 'D-position', 'D_wins', 'round',
           'circuitId', 'name', 'code', 'forename', 'surname'],
          dtype='object')




```python
df_driver_max_points

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
      <th>year</th>
      <th>raceId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
    </tr>
    <tr>
      <th>driverId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2019</td>
      <td>1030</td>
      <td>413.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2018</td>
      <td>1009</td>
      <td>408.0</td>
      <td>2</td>
      <td>11</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>20</td>
      <td>2013</td>
      <td>899</td>
      <td>397.0</td>
      <td>3</td>
      <td>13</td>
      <td>19</td>
      <td>69</td>
      <td>United States Grand Prix</td>
      <td>VET</td>
      <td>Sebastian</td>
      <td>Vettel</td>
    </tr>
    <tr>
      <td>20</td>
      <td>2011</td>
      <td>859</td>
      <td>392.0</td>
      <td>1</td>
      <td>11</td>
      <td>19</td>
      <td>68</td>
      <td>Turkish Grand Prix</td>
      <td>VET</td>
      <td>Sebastian</td>
      <td>Vettel</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016</td>
      <td>968</td>
      <td>385.0</td>
      <td>2</td>
      <td>9</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
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
    </tr>
    <tr>
      <td>411</td>
      <td>1962</td>
      <td>737</td>
      <td>0.0</td>
      <td>37</td>
      <td>0</td>
      <td>9</td>
      <td>56</td>
      <td>South African Grand Prix</td>
      <td>\N</td>
      <td>Doug</td>
      <td>Serrurier</td>
    </tr>
    <tr>
      <td>410</td>
      <td>1965</td>
      <td>708</td>
      <td>0.0</td>
      <td>42</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>United States Grand Prix</td>
      <td>\N</td>
      <td>Neville</td>
      <td>Lederle</td>
    </tr>
    <tr>
      <td>410</td>
      <td>1963</td>
      <td>728</td>
      <td>0.0</td>
      <td>51</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>South African Grand Prix</td>
      <td>\N</td>
      <td>Neville</td>
      <td>Lederle</td>
    </tr>
    <tr>
      <td>409</td>
      <td>1965</td>
      <td>708</td>
      <td>0.0</td>
      <td>40</td>
      <td>0</td>
      <td>10</td>
      <td>56</td>
      <td>United States Grand Prix</td>
      <td>\N</td>
      <td>Trevor</td>
      <td>Blokdyk</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2008</td>
      <td>35</td>
      <td>0.0</td>
      <td>21</td>
      <td>0</td>
      <td>18</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>SAT</td>
      <td>Takuma</td>
      <td>Sato</td>
    </tr>
  </tbody>
</table>
<p>3079 rows × 11 columns</p>
</div>




```python
df_driver_max_points_min=df_driver_max_points.groupby(['driverId']).min()
```


```python
df_driver_max_points_min
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
      <th>year</th>
      <th>raceId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
    </tr>
    <tr>
      <th>driverId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2007</td>
      <td>17</td>
      <td>49.0</td>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2000</td>
      <td>17</td>
      <td>0.0</td>
      <td>5</td>
      <td>0</td>
      <td>16</td>
      <td>18</td>
      <td>Singapore Grand Prix</td>
      <td>HEI</td>
      <td>Nick</td>
      <td>Heidfeld</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2006</td>
      <td>17</td>
      <td>4.0</td>
      <td>2</td>
      <td>0</td>
      <td>17</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2001</td>
      <td>17</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>16</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>ALO</td>
      <td>Fernando</td>
      <td>Alonso</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2007</td>
      <td>17</td>
      <td>0.0</td>
      <td>7</td>
      <td>0</td>
      <td>17</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>KOV</td>
      <td>Heikki</td>
      <td>Kovalainen</td>
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
    </tr>
    <tr>
      <td>844</td>
      <td>2018</td>
      <td>1009</td>
      <td>39.0</td>
      <td>5</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>LEC</td>
      <td>Charles</td>
      <td>Leclerc</td>
    </tr>
    <tr>
      <td>845</td>
      <td>2018</td>
      <td>1009</td>
      <td>1.0</td>
      <td>20</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>SIR</td>
      <td>Sergey</td>
      <td>Sirotkin</td>
    </tr>
    <tr>
      <td>846</td>
      <td>2019</td>
      <td>1030</td>
      <td>49.0</td>
      <td>14</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>NOR</td>
      <td>Lando</td>
      <td>Norris</td>
    </tr>
    <tr>
      <td>847</td>
      <td>2019</td>
      <td>1030</td>
      <td>0.0</td>
      <td>20</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>RUS</td>
      <td>George</td>
      <td>Russell</td>
    </tr>
    <tr>
      <td>848</td>
      <td>2019</td>
      <td>1030</td>
      <td>92.0</td>
      <td>15</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>ALB</td>
      <td>Alexander</td>
      <td>Albon</td>
    </tr>
  </tbody>
</table>
<p>840 rows × 11 columns</p>
</div>




```python
df_driver_max_points_min=df_driver_max_points_min.reset_index()
```


```python
df_driver_max_points_min
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
      <th>driverId</th>
      <th>year</th>
      <th>raceId</th>
      <th>D_points</th>
      <th>D-position</th>
      <th>D_wins</th>
      <th>round</th>
      <th>circuitId</th>
      <th>name</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>17</td>
      <td>49.0</td>
      <td>1</td>
      <td>1</td>
      <td>17</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2000</td>
      <td>17</td>
      <td>0.0</td>
      <td>5</td>
      <td>0</td>
      <td>16</td>
      <td>18</td>
      <td>Singapore Grand Prix</td>
      <td>HEI</td>
      <td>Nick</td>
      <td>Heidfeld</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2006</td>
      <td>17</td>
      <td>4.0</td>
      <td>2</td>
      <td>0</td>
      <td>17</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2001</td>
      <td>17</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>16</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>ALO</td>
      <td>Fernando</td>
      <td>Alonso</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2007</td>
      <td>17</td>
      <td>0.0</td>
      <td>7</td>
      <td>0</td>
      <td>17</td>
      <td>18</td>
      <td>Turkish Grand Prix</td>
      <td>KOV</td>
      <td>Heikki</td>
      <td>Kovalainen</td>
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
    </tr>
    <tr>
      <td>835</td>
      <td>844</td>
      <td>2018</td>
      <td>1009</td>
      <td>39.0</td>
      <td>5</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>LEC</td>
      <td>Charles</td>
      <td>Leclerc</td>
    </tr>
    <tr>
      <td>836</td>
      <td>845</td>
      <td>2018</td>
      <td>1009</td>
      <td>1.0</td>
      <td>20</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>SIR</td>
      <td>Sergey</td>
      <td>Sirotkin</td>
    </tr>
    <tr>
      <td>837</td>
      <td>846</td>
      <td>2019</td>
      <td>1030</td>
      <td>49.0</td>
      <td>14</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>NOR</td>
      <td>Lando</td>
      <td>Norris</td>
    </tr>
    <tr>
      <td>838</td>
      <td>847</td>
      <td>2019</td>
      <td>1030</td>
      <td>0.0</td>
      <td>20</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>RUS</td>
      <td>George</td>
      <td>Russell</td>
    </tr>
    <tr>
      <td>839</td>
      <td>848</td>
      <td>2019</td>
      <td>1030</td>
      <td>92.0</td>
      <td>15</td>
      <td>0</td>
      <td>21</td>
      <td>73</td>
      <td>United States Grand Prix</td>
      <td>ALB</td>
      <td>Alexander</td>
      <td>Albon</td>
    </tr>
  </tbody>
</table>
<p>840 rows × 12 columns</p>
</div>




```python
df_driver_max_points_min=df_driver_max_points_min[['driverId','year']]
```


```python
df_driver_max_points_min.columns=['driverId', 'start_year']
```


```python
df_driver_max_points_min.columns
```




    Index(['driverId', 'start_year'], dtype='object')




```python
df_driver_max_points_max=df_driver_max_points.groupby(['driverId']).max()
```


```python
df_driver_max_points_max=df_driver_max_points_max.reset_index()
```


```python
df_driver_max_points_max=df_driver_max_points_max[['driverId','year']]
```


```python
df_driver_max_points_max.columns=['driverId', 'finish_year']
```


```python
df_driver_max_points_max.columns
```




    Index(['driverId', 'finish_year'], dtype='object')




```python
df_driver_raceperiod=pd.merge(df_driver_max_points_min,df_driver_max_points_max,how='inner',on='driverId')
```


```python
df_driver_raceperiod
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
      <th>driverId</th>
      <th>start_year</th>
      <th>finish_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2000</td>
      <td>2011</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2006</td>
      <td>2016</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2001</td>
      <td>2018</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2007</td>
      <td>2013</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>835</td>
      <td>844</td>
      <td>2018</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>836</td>
      <td>845</td>
      <td>2018</td>
      <td>2018</td>
    </tr>
    <tr>
      <td>837</td>
      <td>846</td>
      <td>2019</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>838</td>
      <td>847</td>
      <td>2019</td>
      <td>2019</td>
    </tr>
    <tr>
      <td>839</td>
      <td>848</td>
      <td>2019</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
<p>840 rows × 3 columns</p>
</div>




```python
df_driver_raceperiod=df_driver_raceperiod.assign(racing_time=" ")
```


```python
df_driver_raceperiod['racing_time']=df_driver_raceperiod['finish_year']-df_driver_raceperiod['start_year']
```


```python
df_driver_raceperiod['racing_time']=df_driver_raceperiod['racing_time']+1
```


```python
df_driver_raceperiod.to_csv('df_driver_raceperiod.csv')
```


```python
df_driver_raceperiod_analysis1=df_driver_raceperiod.loc[(df_driver_raceperiod['finish_year']>=2018)]
```


```python
df_driver_raceperiod_analysis1.sort_values(by='racing_time',ascending=False)
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
      <th>driverId</th>
      <th>start_year</th>
      <th>finish_year</th>
      <th>racing_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>2001</td>
      <td>2019</td>
      <td>19</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2001</td>
      <td>2018</td>
      <td>18</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>2006</td>
      <td>2019</td>
      <td>14</td>
    </tr>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>2019</td>
      <td>13</td>
    </tr>
    <tr>
      <td>19</td>
      <td>20</td>
      <td>2007</td>
      <td>2019</td>
      <td>13</td>
    </tr>
    <tr>
      <td>146</td>
      <td>154</td>
      <td>2009</td>
      <td>2019</td>
      <td>11</td>
    </tr>
    <tr>
      <td>799</td>
      <td>807</td>
      <td>2010</td>
      <td>2019</td>
      <td>10</td>
    </tr>
    <tr>
      <td>806</td>
      <td>815</td>
      <td>2011</td>
      <td>2019</td>
      <td>9</td>
    </tr>
    <tr>
      <td>808</td>
      <td>817</td>
      <td>2011</td>
      <td>2019</td>
      <td>9</td>
    </tr>
    <tr>
      <td>813</td>
      <td>822</td>
      <td>2013</td>
      <td>2019</td>
      <td>7</td>
    </tr>
    <tr>
      <td>816</td>
      <td>825</td>
      <td>2014</td>
      <td>2019</td>
      <td>6</td>
    </tr>
    <tr>
      <td>817</td>
      <td>826</td>
      <td>2014</td>
      <td>2019</td>
      <td>6</td>
    </tr>
    <tr>
      <td>823</td>
      <td>832</td>
      <td>2015</td>
      <td>2019</td>
      <td>5</td>
    </tr>
    <tr>
      <td>821</td>
      <td>830</td>
      <td>2015</td>
      <td>2019</td>
      <td>5</td>
    </tr>
    <tr>
      <td>819</td>
      <td>828</td>
      <td>2014</td>
      <td>2018</td>
      <td>5</td>
    </tr>
    <tr>
      <td>829</td>
      <td>838</td>
      <td>2016</td>
      <td>2018</td>
      <td>3</td>
    </tr>
    <tr>
      <td>830</td>
      <td>839</td>
      <td>2016</td>
      <td>2018</td>
      <td>3</td>
    </tr>
    <tr>
      <td>831</td>
      <td>840</td>
      <td>2017</td>
      <td>2019</td>
      <td>3</td>
    </tr>
    <tr>
      <td>832</td>
      <td>841</td>
      <td>2017</td>
      <td>2019</td>
      <td>3</td>
    </tr>
    <tr>
      <td>833</td>
      <td>842</td>
      <td>2017</td>
      <td>2019</td>
      <td>3</td>
    </tr>
    <tr>
      <td>834</td>
      <td>843</td>
      <td>2017</td>
      <td>2018</td>
      <td>2</td>
    </tr>
    <tr>
      <td>835</td>
      <td>844</td>
      <td>2018</td>
      <td>2019</td>
      <td>2</td>
    </tr>
    <tr>
      <td>836</td>
      <td>845</td>
      <td>2018</td>
      <td>2018</td>
      <td>1</td>
    </tr>
    <tr>
      <td>837</td>
      <td>846</td>
      <td>2019</td>
      <td>2019</td>
      <td>1</td>
    </tr>
    <tr>
      <td>838</td>
      <td>847</td>
      <td>2019</td>
      <td>2019</td>
      <td>1</td>
    </tr>
    <tr>
      <td>839</td>
      <td>848</td>
      <td>2019</td>
      <td>2019</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_driver_raceperiod_analysis1.to_csv('./intermediate_files/df_driver_raceperiod_analysis1.csv')
```


```python
df_driver_raceperiod_analysis1=df_driver_raceperiod_analysis1.loc[(df_driver_raceperiod_analysis1['finish_year']>=2018)]
```


```python
# calculating racing experience for a racer

df_driver_raceperiod_analysis1.sort_values(by='racing_time',ascending=False)
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
      <th>driverId</th>
      <th>start_year</th>
      <th>finish_year</th>
      <th>racing_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>2001</td>
      <td>2019</td>
      <td>19</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2001</td>
      <td>2018</td>
      <td>18</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>2006</td>
      <td>2019</td>
      <td>14</td>
    </tr>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>2019</td>
      <td>13</td>
    </tr>
    <tr>
      <td>19</td>
      <td>20</td>
      <td>2007</td>
      <td>2019</td>
      <td>13</td>
    </tr>
    <tr>
      <td>146</td>
      <td>154</td>
      <td>2009</td>
      <td>2019</td>
      <td>11</td>
    </tr>
    <tr>
      <td>799</td>
      <td>807</td>
      <td>2010</td>
      <td>2019</td>
      <td>10</td>
    </tr>
    <tr>
      <td>806</td>
      <td>815</td>
      <td>2011</td>
      <td>2019</td>
      <td>9</td>
    </tr>
    <tr>
      <td>808</td>
      <td>817</td>
      <td>2011</td>
      <td>2019</td>
      <td>9</td>
    </tr>
    <tr>
      <td>813</td>
      <td>822</td>
      <td>2013</td>
      <td>2019</td>
      <td>7</td>
    </tr>
    <tr>
      <td>816</td>
      <td>825</td>
      <td>2014</td>
      <td>2019</td>
      <td>6</td>
    </tr>
    <tr>
      <td>817</td>
      <td>826</td>
      <td>2014</td>
      <td>2019</td>
      <td>6</td>
    </tr>
    <tr>
      <td>823</td>
      <td>832</td>
      <td>2015</td>
      <td>2019</td>
      <td>5</td>
    </tr>
    <tr>
      <td>821</td>
      <td>830</td>
      <td>2015</td>
      <td>2019</td>
      <td>5</td>
    </tr>
    <tr>
      <td>819</td>
      <td>828</td>
      <td>2014</td>
      <td>2018</td>
      <td>5</td>
    </tr>
    <tr>
      <td>829</td>
      <td>838</td>
      <td>2016</td>
      <td>2018</td>
      <td>3</td>
    </tr>
    <tr>
      <td>830</td>
      <td>839</td>
      <td>2016</td>
      <td>2018</td>
      <td>3</td>
    </tr>
    <tr>
      <td>831</td>
      <td>840</td>
      <td>2017</td>
      <td>2019</td>
      <td>3</td>
    </tr>
    <tr>
      <td>832</td>
      <td>841</td>
      <td>2017</td>
      <td>2019</td>
      <td>3</td>
    </tr>
    <tr>
      <td>833</td>
      <td>842</td>
      <td>2017</td>
      <td>2019</td>
      <td>3</td>
    </tr>
    <tr>
      <td>834</td>
      <td>843</td>
      <td>2017</td>
      <td>2018</td>
      <td>2</td>
    </tr>
    <tr>
      <td>835</td>
      <td>844</td>
      <td>2018</td>
      <td>2019</td>
      <td>2</td>
    </tr>
    <tr>
      <td>836</td>
      <td>845</td>
      <td>2018</td>
      <td>2018</td>
      <td>1</td>
    </tr>
    <tr>
      <td>837</td>
      <td>846</td>
      <td>2019</td>
      <td>2019</td>
      <td>1</td>
    </tr>
    <tr>
      <td>838</td>
      <td>847</td>
      <td>2019</td>
      <td>2019</td>
      <td>1</td>
    </tr>
    <tr>
      <td>839</td>
      <td>848</td>
      <td>2019</td>
      <td>2019</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_driver_raceperiod_analysis_2=df_driver_raceperiod.loc[(df_driver_raceperiod['start_year']>=1999)&(df_driver_raceperiod['finish_year']<=2019)]
```


```python
df_driver_raceperiod_analysis_2.sort_values(by='racing_time',ascending=False).head(30)
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
      <th>driverId</th>
      <th>start_year</th>
      <th>finish_year</th>
      <th>racing_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>2001</td>
      <td>2019</td>
      <td>19</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>2001</td>
      <td>2018</td>
      <td>18</td>
    </tr>
    <tr>
      <td>17</td>
      <td>18</td>
      <td>2000</td>
      <td>2017</td>
      <td>18</td>
    </tr>
    <tr>
      <td>12</td>
      <td>13</td>
      <td>2002</td>
      <td>2017</td>
      <td>16</td>
    </tr>
    <tr>
      <td>35</td>
      <td>37</td>
      <td>1999</td>
      <td>2012</td>
      <td>14</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>2006</td>
      <td>2019</td>
      <td>14</td>
    </tr>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2007</td>
      <td>2019</td>
      <td>13</td>
    </tr>
    <tr>
      <td>19</td>
      <td>20</td>
      <td>2007</td>
      <td>2019</td>
      <td>13</td>
    </tr>
    <tr>
      <td>16</td>
      <td>17</td>
      <td>2002</td>
      <td>2013</td>
      <td>12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2000</td>
      <td>2011</td>
      <td>12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>2006</td>
      <td>2016</td>
      <td>11</td>
    </tr>
    <tr>
      <td>146</td>
      <td>154</td>
      <td>2009</td>
      <td>2019</td>
      <td>11</td>
    </tr>
    <tr>
      <td>799</td>
      <td>807</td>
      <td>2010</td>
      <td>2019</td>
      <td>10</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>2004</td>
      <td>2012</td>
      <td>9</td>
    </tr>
    <tr>
      <td>806</td>
      <td>815</td>
      <td>2011</td>
      <td>2019</td>
      <td>9</td>
    </tr>
    <tr>
      <td>808</td>
      <td>817</td>
      <td>2011</td>
      <td>2019</td>
      <td>9</td>
    </tr>
    <tr>
      <td>15</td>
      <td>16</td>
      <td>2007</td>
      <td>2014</td>
      <td>8</td>
    </tr>
    <tr>
      <td>37</td>
      <td>39</td>
      <td>2005</td>
      <td>2012</td>
      <td>8</td>
    </tr>
    <tr>
      <td>30</td>
      <td>32</td>
      <td>2004</td>
      <td>2010</td>
      <td>7</td>
    </tr>
    <tr>
      <td>805</td>
      <td>814</td>
      <td>2011</td>
      <td>2017</td>
      <td>7</td>
    </tr>
    <tr>
      <td>813</td>
      <td>822</td>
      <td>2013</td>
      <td>2019</td>
      <td>7</td>
    </tr>
    <tr>
      <td>10</td>
      <td>11</td>
      <td>2002</td>
      <td>2008</td>
      <td>7</td>
    </tr>
    <tr>
      <td>23</td>
      <td>24</td>
      <td>2005</td>
      <td>2011</td>
      <td>7</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>2007</td>
      <td>2013</td>
      <td>7</td>
    </tr>
    <tr>
      <td>29</td>
      <td>31</td>
      <td>2001</td>
      <td>2006</td>
      <td>6</td>
    </tr>
    <tr>
      <td>816</td>
      <td>825</td>
      <td>2014</td>
      <td>2019</td>
      <td>6</td>
    </tr>
    <tr>
      <td>147</td>
      <td>155</td>
      <td>2009</td>
      <td>2014</td>
      <td>6</td>
    </tr>
    <tr>
      <td>817</td>
      <td>826</td>
      <td>2014</td>
      <td>2019</td>
      <td>6</td>
    </tr>
    <tr>
      <td>39</td>
      <td>41</td>
      <td>1999</td>
      <td>2004</td>
      <td>6</td>
    </tr>
    <tr>
      <td>46</td>
      <td>48</td>
      <td>1999</td>
      <td>2004</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_Highest_grand_prix_entered=pd.read_csv('./intermediate_files//Highest_grand_prix_entered.csv',index_col=0)
```


```python
df_lifetime_winning=pd.read_csv('./intermediate_files/Lifetime_wins.csv',index_col=0)
```


```python
df_percentagewinning=pd.merge(df_Highest_grand_prix_entered,df_lifetime_winning,how='inner',on='driverId')
```


```python
df_percentagewinning=df_percentagewinning[['TotalRacesplayed','forename_x','surname_x','driverId','LifeTimeWinning']]
```


```python
df_percentagewinning.assign(percentage_winning= " ")
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
      <th>TotalRacesplayed</th>
      <th>forename_x</th>
      <th>surname_x</th>
      <th>driverId</th>
      <th>LifeTimeWinning</th>
      <th>percentage_winning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>316</td>
      <td>Jenson</td>
      <td>Button</td>
      <td>18</td>
      <td>15</td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>316</td>
      <td>Fernando</td>
      <td>Alonso</td>
      <td>4</td>
      <td>32</td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>314</td>
      <td>Kimi</td>
      <td>Raikkonen</td>
      <td>8</td>
      <td>21</td>
      <td></td>
    </tr>
    <tr>
      <td>3</td>
      <td>312</td>
      <td>Michael</td>
      <td>Schumacher</td>
      <td>30</td>
      <td>91</td>
      <td></td>
    </tr>
    <tr>
      <td>4</td>
      <td>308</td>
      <td>Rubens</td>
      <td>Barrichello</td>
      <td>22</td>
      <td>11</td>
      <td></td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>835</td>
      <td>1</td>
      <td>Alberto</td>
      <td>Crespo</td>
      <td>761</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <td>836</td>
      <td>1</td>
      <td>Juan</td>
      <td>Jover</td>
      <td>782</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <td>837</td>
      <td>1</td>
      <td>Georges</td>
      <td>Grignard</td>
      <td>783</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <td>838</td>
      <td>1</td>
      <td>Dorino</td>
      <td>Serafini</td>
      <td>802</td>
      <td>0</td>
      <td></td>
    </tr>
    <tr>
      <td>839</td>
      <td>1</td>
      <td>Phil</td>
      <td>Cade</td>
      <td>576</td>
      <td>0</td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>840 rows × 6 columns</p>
</div>




```python
df_percentagewinning['percentage_winning']=(df_percentagewinning['LifeTimeWinning']/df_percentagewinning['TotalRacesplayed'])*100
```


```python
df_percentagewinning
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
      <th>TotalRacesplayed</th>
      <th>forename_x</th>
      <th>surname_x</th>
      <th>driverId</th>
      <th>LifeTimeWinning</th>
      <th>percentage_winning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>316</td>
      <td>Jenson</td>
      <td>Button</td>
      <td>18</td>
      <td>15</td>
      <td>4.746835</td>
    </tr>
    <tr>
      <td>1</td>
      <td>316</td>
      <td>Fernando</td>
      <td>Alonso</td>
      <td>4</td>
      <td>32</td>
      <td>10.126582</td>
    </tr>
    <tr>
      <td>2</td>
      <td>314</td>
      <td>Kimi</td>
      <td>Raikkonen</td>
      <td>8</td>
      <td>21</td>
      <td>6.687898</td>
    </tr>
    <tr>
      <td>3</td>
      <td>312</td>
      <td>Michael</td>
      <td>Schumacher</td>
      <td>30</td>
      <td>91</td>
      <td>29.166667</td>
    </tr>
    <tr>
      <td>4</td>
      <td>308</td>
      <td>Rubens</td>
      <td>Barrichello</td>
      <td>22</td>
      <td>11</td>
      <td>3.571429</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>835</td>
      <td>1</td>
      <td>Alberto</td>
      <td>Crespo</td>
      <td>761</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>836</td>
      <td>1</td>
      <td>Juan</td>
      <td>Jover</td>
      <td>782</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>837</td>
      <td>1</td>
      <td>Georges</td>
      <td>Grignard</td>
      <td>783</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>838</td>
      <td>1</td>
      <td>Dorino</td>
      <td>Serafini</td>
      <td>802</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>839</td>
      <td>1</td>
      <td>Phil</td>
      <td>Cade</td>
      <td>576</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>840 rows × 6 columns</p>
</div>




```python
#df_driver_max_points=df_driver_max_points[['forename','percentage_winning']]
```


```python
df_percentagewinning=df_percentagewinning.sort_values(by='percentage_winning',ascending=False)
```


```python
df_percentagewinning
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
      <th>TotalRacesplayed</th>
      <th>forename_x</th>
      <th>surname_x</th>
      <th>driverId</th>
      <th>LifeTimeWinning</th>
      <th>percentage_winning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>137</td>
      <td>67</td>
      <td>Juan</td>
      <td>Fangio</td>
      <td>579</td>
      <td>24</td>
      <td>35.820896</td>
    </tr>
    <tr>
      <td>8</td>
      <td>249</td>
      <td>Lewis</td>
      <td>Hamilton</td>
      <td>1</td>
      <td>84</td>
      <td>33.734940</td>
    </tr>
    <tr>
      <td>217</td>
      <td>43</td>
      <td>Alberto</td>
      <td>Ascari</td>
      <td>647</td>
      <td>13</td>
      <td>30.232558</td>
    </tr>
    <tr>
      <td>3</td>
      <td>312</td>
      <td>Michael</td>
      <td>Schumacher</td>
      <td>30</td>
      <td>91</td>
      <td>29.166667</td>
    </tr>
    <tr>
      <td>108</td>
      <td>86</td>
      <td>Jim</td>
      <td>Clark</td>
      <td>373</td>
      <td>25</td>
      <td>29.069767</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>351</td>
      <td>25</td>
      <td>Luciano</td>
      <td>Burti</td>
      <td>54</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>352</td>
      <td>25</td>
      <td>Antônio</td>
      <td>Pizzonia</td>
      <td>42</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>353</td>
      <td>24</td>
      <td>Len</td>
      <td>Sutton</td>
      <td>536</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>354</td>
      <td>24</td>
      <td>Bruce</td>
      <td>Halford</td>
      <td>506</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>839</td>
      <td>1</td>
      <td>Phil</td>
      <td>Cade</td>
      <td>576</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>840 rows × 6 columns</p>
</div>




```python
df_percentagewinning.to_csv('percentage_winning1.csv')
```


```python
df_percentagewinning.columns
```




    Index(['TotalRacesplayed', 'forename_x', 'surname_x', 'driverId',
           'LifeTimeWinning', 'percentage_winning'],
          dtype='object')




```python
plt.figure(figsize=(10,5))
sns.barplot(df_percentagewinning.forename_x.head(10),df_percentagewinning.percentage_winning.head(10),alpha=1)
plt.xticks(fontsize=22,rotation=45)
plt.title('Winning Percenage of the Drivers',fontsize=20)
plt.ylabel('Winning percentage', fontsize=20)
plt.xlabel('Driver First name', fontsize=22)
plt.show()
```


![png](output_139_0.png)


### *** Findings *** ###

***Juan*** has highest winning percentage than ***Lewis***. Currently he is not an active racer.His last race was at 2000.Hence, ***Lewis is the world champion***.

### Driver Details for age ###


```python
df_driver=pd.read_csv('./Capstone_core_data/drivers.csv')
```


```python
df_driver.head()
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
      <th>driverId</th>
      <th>driverRef</th>
      <th>number</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
      <th>dob</th>
      <th>nationality</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>hamilton</td>
      <td>44</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
      <td>1985-01-07</td>
      <td>British</td>
      <td>http://en.wikipedia.org/wiki/Lewis_Hamilton</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>heidfeld</td>
      <td>\N</td>
      <td>HEI</td>
      <td>Nick</td>
      <td>Heidfeld</td>
      <td>1977-05-10</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nick_Heidfeld</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>rosberg</td>
      <td>6</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
      <td>1985-06-27</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nico_Rosberg</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>alonso</td>
      <td>14</td>
      <td>ALO</td>
      <td>Fernando</td>
      <td>Alonso</td>
      <td>1981-07-29</td>
      <td>Spanish</td>
      <td>http://en.wikipedia.org/wiki/Fernando_Alonso</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>kovalainen</td>
      <td>\N</td>
      <td>KOV</td>
      <td>Heikki</td>
      <td>Kovalainen</td>
      <td>1981-10-19</td>
      <td>Finnish</td>
      <td>http://en.wikipedia.org/wiki/Heikki_Kovalainen</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_driver.columns=['driverId','driverRef','number','code','forename','surname','dob','nationality','url']
```


```python
df_driver.assign(age= " ")
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
      <th>driverId</th>
      <th>driverRef</th>
      <th>number</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
      <th>dob</th>
      <th>nationality</th>
      <th>url</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>hamilton</td>
      <td>44</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
      <td>1985-01-07</td>
      <td>British</td>
      <td>http://en.wikipedia.org/wiki/Lewis_Hamilton</td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>heidfeld</td>
      <td>\N</td>
      <td>HEI</td>
      <td>Nick</td>
      <td>Heidfeld</td>
      <td>1977-05-10</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nick_Heidfeld</td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>rosberg</td>
      <td>6</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
      <td>1985-06-27</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nico_Rosberg</td>
      <td></td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>alonso</td>
      <td>14</td>
      <td>ALO</td>
      <td>Fernando</td>
      <td>Alonso</td>
      <td>1981-07-29</td>
      <td>Spanish</td>
      <td>http://en.wikipedia.org/wiki/Fernando_Alonso</td>
      <td></td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>kovalainen</td>
      <td>\N</td>
      <td>KOV</td>
      <td>Heikki</td>
      <td>Kovalainen</td>
      <td>1981-10-19</td>
      <td>Finnish</td>
      <td>http://en.wikipedia.org/wiki/Heikki_Kovalainen</td>
      <td></td>
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
    </tr>
    <tr>
      <td>842</td>
      <td>844</td>
      <td>leclerc</td>
      <td>16</td>
      <td>LEC</td>
      <td>Charles</td>
      <td>Leclerc</td>
      <td>1997-10-16</td>
      <td>Monegasque</td>
      <td>http://en.wikipedia.org/wiki/Charles_Leclerc</td>
      <td></td>
    </tr>
    <tr>
      <td>843</td>
      <td>845</td>
      <td>sirotkin</td>
      <td>35</td>
      <td>SIR</td>
      <td>Sergey</td>
      <td>Sirotkin</td>
      <td>1995-08-27</td>
      <td>Russian</td>
      <td>http://en.wikipedia.org/wiki/Sergey_Sirotkin_(...</td>
      <td></td>
    </tr>
    <tr>
      <td>844</td>
      <td>846</td>
      <td>norris</td>
      <td>4</td>
      <td>NOR</td>
      <td>Lando</td>
      <td>Norris</td>
      <td>1999-11-13</td>
      <td>British</td>
      <td>http://en.wikipedia.org/wiki/Lando_Norris</td>
      <td></td>
    </tr>
    <tr>
      <td>845</td>
      <td>847</td>
      <td>russell</td>
      <td>63</td>
      <td>RUS</td>
      <td>George</td>
      <td>Russell</td>
      <td>1998-02-15</td>
      <td>British</td>
      <td>http://en.wikipedia.org/wiki/George_Russell_(r...</td>
      <td></td>
    </tr>
    <tr>
      <td>846</td>
      <td>848</td>
      <td>albon</td>
      <td>23</td>
      <td>ALB</td>
      <td>Alexander</td>
      <td>Albon</td>
      <td>1996-03-23</td>
      <td>Thai</td>
      <td>http://en.wikipedia.org/wiki/Alexander_Albon</td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>847 rows × 10 columns</p>
</div>




```python
import pandas as pd
from datetime import datetime
from datetime import date

d = df_driver['dob']

df = pd.DataFrame(data=d)

def calculate_age(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df_driver['age'] = df_driver['dob'].apply(calculate_age)
df_driver['age'].head()
```




    0    35
    1    42
    2    34
    3    38
    4    38
    Name: age, dtype: int64



### Select driver ID,age from driver.csv


```python
df_driver.head(10)
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
      <th>driverId</th>
      <th>driverRef</th>
      <th>number</th>
      <th>code</th>
      <th>forename</th>
      <th>surname</th>
      <th>dob</th>
      <th>nationality</th>
      <th>url</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>hamilton</td>
      <td>44</td>
      <td>HAM</td>
      <td>Lewis</td>
      <td>Hamilton</td>
      <td>1985-01-07</td>
      <td>British</td>
      <td>http://en.wikipedia.org/wiki/Lewis_Hamilton</td>
      <td>35</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>heidfeld</td>
      <td>\N</td>
      <td>HEI</td>
      <td>Nick</td>
      <td>Heidfeld</td>
      <td>1977-05-10</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nick_Heidfeld</td>
      <td>42</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>rosberg</td>
      <td>6</td>
      <td>ROS</td>
      <td>Nico</td>
      <td>Rosberg</td>
      <td>1985-06-27</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Nico_Rosberg</td>
      <td>34</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>alonso</td>
      <td>14</td>
      <td>ALO</td>
      <td>Fernando</td>
      <td>Alonso</td>
      <td>1981-07-29</td>
      <td>Spanish</td>
      <td>http://en.wikipedia.org/wiki/Fernando_Alonso</td>
      <td>38</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>kovalainen</td>
      <td>\N</td>
      <td>KOV</td>
      <td>Heikki</td>
      <td>Kovalainen</td>
      <td>1981-10-19</td>
      <td>Finnish</td>
      <td>http://en.wikipedia.org/wiki/Heikki_Kovalainen</td>
      <td>38</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>nakajima</td>
      <td>\N</td>
      <td>NAK</td>
      <td>Kazuki</td>
      <td>Nakajima</td>
      <td>1985-01-11</td>
      <td>Japanese</td>
      <td>http://en.wikipedia.org/wiki/Kazuki_Nakajima</td>
      <td>35</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>bourdais</td>
      <td>\N</td>
      <td>BOU</td>
      <td>Sébastien</td>
      <td>Bourdais</td>
      <td>1979-02-28</td>
      <td>French</td>
      <td>http://en.wikipedia.org/wiki/S%C3%A9bastien_Bo...</td>
      <td>41</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>raikkonen</td>
      <td>7</td>
      <td>RAI</td>
      <td>Kimi</td>
      <td>Raikkonen</td>
      <td>1979-10-17</td>
      <td>Finnish</td>
      <td>http://en.wikipedia.org/wiki/Kimi_R%C3%A4ikk%C...</td>
      <td>40</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>kubica</td>
      <td>88</td>
      <td>KUB</td>
      <td>Robert</td>
      <td>Kubica</td>
      <td>1984-12-07</td>
      <td>Polish</td>
      <td>http://en.wikipedia.org/wiki/Robert_Kubica</td>
      <td>35</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>glock</td>
      <td>\N</td>
      <td>GLO</td>
      <td>Timo</td>
      <td>Glock</td>
      <td>1982-03-18</td>
      <td>German</td>
      <td>http://en.wikipedia.org/wiki/Timo_Glock</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_Driver_Analysis=df_driver[['driverId','age','nationality']]

```


```python
df_Driver_Analysis.to_csv("driver_age.csv")
```


```python
df_Driver_Analysis=df_Driver_Analysis[['driverId','nationality']]
```


```python
#df_race_played=df_driver_standings.groupby(['driverId']).raceId.count().reset_index().rename(columns={"raceId":"TotalRacesplayed"})
df_analysis=df_Driver_Analysis.groupby(['nationality']).count()
```


```python
df_analysis=df_analysis.reset_index()
```


```python
df_analysis=df_analysis.rename(columns={"nationality":"Nationality","driverId":"Count_from_each_country"})
```


```python
df_analysis.columns
```




    Index(['Nationality', 'Count_from_each_country'], dtype='object')




```python
df_analysis=df_analysis.sort_values(by='Count_from_each_country',ascending=False).head(10)
```


```python
df_analysis.head(10)
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
      <th>Nationality</th>
      <th>Count_from_each_country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8</td>
      <td>British</td>
      <td>164</td>
    </tr>
    <tr>
      <td>0</td>
      <td>American</td>
      <td>157</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Italian</td>
      <td>99</td>
    </tr>
    <tr>
      <td>17</td>
      <td>French</td>
      <td>73</td>
    </tr>
    <tr>
      <td>18</td>
      <td>German</td>
      <td>49</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Brazilian</td>
      <td>31</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Argentine</td>
      <td>24</td>
    </tr>
    <tr>
      <td>37</td>
      <td>Swiss</td>
      <td>23</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Belgian</td>
      <td>23</td>
    </tr>
    <tr>
      <td>34</td>
      <td>South African</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(20,30))
sns.barplot(df_analysis.head(5).Nationality, df_analysis.head(5).Count_from_each_country, alpha=1)
plt.title('Total drivers from each country in the Formula One Race')
plt.ylabel('Counts in Races', fontsize=12)
plt.xlabel('Nationality', fontsize=12)
plt.show()
```


![png](output_159_0.png)


### ***Findings*** ###


Race demographics shows the most of the drivers are from Britan.Next to British drivers, americans are the leading drivers.Top 5 demographics from findings:

    1.Britan
    2.America
    3.Italy
    4.France
    5.German

Lewis Hamilton is a British Driver

---
End of the the analysis

---

---
End of the notebook

---
