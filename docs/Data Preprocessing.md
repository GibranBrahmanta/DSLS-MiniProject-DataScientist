# Data Preprocessing

Overall Data Preprocessing Flow

![Overall Data Preprocessing Flow](assets/dataprep_flow.png "Overall Data Preprocessing Flow")

## Jams Dataset 

Jams Data Preprocessing Flow

![Jams Data Preprocessing Flow](assets/jams_preprop.png "Jams Data Preprocessing Flow")

Jams dataset are being processed by using these following steps:

1.  Remove rows that has duplicate (street, timestamp) data
    * Note: If there were duplication, keep row that has highest vote count 
2.  Filter used street based on completion rate
    * Used street are streets that has completion rate >= 90 percentile of completion rate distribution
3.  Compute the coordinate of each used street by calculating the mean value of latitude and longitude on each related data 
    * Result from this step will be used later on to create the weather dataset    
4.  On each street that are being used, create data for all of those timestamps that doesn't appear in the original dataset in order to create a full historical data. Below are the description on each used attributes on the new created data
    * current_timestamp: new timestamp that doesn't appear in the original data
    * street: name of the street
    * level: 0, by using assumption that a timestamp that doesn't appear in the original means that there were no traffic jam on that time
    * median_speed: computed by using the domain knowledge about the speed comparison on each jam level. Randomized value also being included on this step to create more variative dataset 
    * median_length: computed by using the comparison between median_speed and median_length on the original dataset 
    * median_delay: computed by using the comparison between median_speed and median_delay on the original dataset

## Irregularities Dataset

Irregularities Data Preprocessing Flow

![Irregularities Data Preprocessing Flow](assets/Irregularities_preprop.png "Irregularities Data Preprocessing Flow")

Irregularities dataset are being processed by using these following steps:

1.  Remove rows that has duplicate (street, timestamp) data
    * Note: If there were duplication, keep row that has highest vote count
2.  Filter used street based on Cleaned Jams Data
3.  On each street that are being used, create data for all of those timestamps that doesn't appear in the original dataset in order to create a full historical data. Below are the description on each used attributes on the new created data
    * current_timestamp: new timestamp that doesn't appear in the original data
    * street: name of the street
    * median_regular_speed: computed by searching the median value of 'median_speed' in completed jams dataset that related to the processed street and has timestamp before or equal the processed timestamp
    * median_delay_seconds: computed by searching the median value of 'median_delay' in completed jams dataset that related to the processed street and has timestamp before or equal the processed timestamp

## Weather Dataset

Weather Data Preprocessing Flow

![Weather Data Preprocessing Flow](assets/Weather_preprop.png "Weather Data Preprocessing Flow")

Weather dataset are being processed by using these following steps:

1. By using the coordinate data that has been got from jams dataset, search for the mean value of latitude and longitude between all of the used street
    * These method were used in order to reduce the number of API Call that has to be made to get the weather data. The number of API Call has to be limited due to lack of resource (money, because the API has limited number of free request)
2. Generate all possible timestamp based on start timestamp and end timestamp that are being determined by the config file
3. For each timestamp, get the weather dataset using the [OpenWeather](https://openweathermap.org/api/one-call-3) API

## Holiday Dataset 

Holiday Data Preprocessing Flow

![Holiday Data Preprocessing Flow](assets/Holiday_preprop.png "Holiday Data Preprocessing Flow")

Holiday dataset are being processed by using these following step:

1. Scrape all of the holiday data on the year that has been determined by the config file through this [page](https://excelnotes.com/holidays-indonesia-2022/)
2. Create the table format of the data

## Merge Dataset

![Merge Data Flow](assets/merge_preprop.png "Merge Data Flow")

All dataset that has been described before are being merged into one dataset by these following steps:

1. Jams and Ireregularities dataset are being joined based on (timestamp, street)
2. Weather data is being merged with the result from previous step based on timestamp
3. Holiday data is being mereged with the result from preivous step by computing the day gap between the timestamp and the nearest holiday date. The result will has range mininum -1 and maxinum 7 where it tells how many days before/after the nearest holiday on a timestamp. -1 if current timestamp doesn't have any nearest holiday in a one week time span
4. Level are being grouped into several group in order to reduce the imbalanced
   * Level grouping
     * Group 0: Low traffic jam (level 0-1)
     * Group 1: Medium traffic jam (level 2-3)
     * Group 2: High traffic jam (level 4-5)
5. Several additional attributes such as 'time_series_split' and 'classification_split' also being created in order to indicate the role of each row on the modeling (train/valid/test)