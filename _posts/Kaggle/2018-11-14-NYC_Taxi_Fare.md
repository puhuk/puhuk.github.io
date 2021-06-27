---
title: "Kaggle - NYC taxi fare description (1)"
date: 2018-11-15 08:26:28 -0400
categories: Kaggle practice
use_math: true
---

As I study machine learning and data science, I have interest to solving Kaggle problem. My goal with Kaggle is to improve my data science knowledge and skill. Before this, my first object is to be accustomed to Kaggle and this kind of problem solving. So firstly, I will analyze how others win and get better score.
[NYC taxi fare prediction competition](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) is the first problem from Kaggle that I want to study.
The goal of this competition is to predict NYC taxi fare, and here in this post, will share how others solved and what I learned from.

### Overview
#### Data
pickup_datetime : When the taxi ride started (timestamp)
pickup_longitude : Longtitude coordinate of taxi ride started (float)
pickup_latitude : Latitude coordinate of taxi ride started (float)
dropoff_longtitude : Longtitude coordinate of taxi ride ended (float)
dropoff_latitude : Latitude coordinate of taxi ride ended (float)
passenger_count : Number of passengers (integer)
#### Evaluation
Using RMSE to measure the difference between the predictions and the fare.
$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2}
$$

So here, the lower RMSE means better prediction.

### Idea
I studied with [Sylas's kernel](https://www.kaggle.com/jsylas/top-ten-rank-r-22m-rows-2-90-lightgbm), and share the overall logic of this kernel.

There should be some pre-knowledge before understanding this kernel.
- Latitude is between -90 and 90, while longitude between -180 and 180 in degree.
- For 2 radian coordinates $$$(a_1,b_1), (a_2,b_2)$$$ in sphere with radius $$$r$$$, the distance of 2 points is $$$2*r*asin\sqrt{(sin(\frac{\Delta a}{2})^2+cos(a_1)*cos(a_2)*sin(\frac{\Delta b}{2})^2)}$$$
- To change the latitude or longitude to radian coordinate from degree coordinate, following formula should be adapted $$$Radian=Degree*\pi/180$$$
- There can be layover between pickup and dropoff, so the kernel added few spots assumed to be stopped by. (JFK, Liberty statue, etc.)

### Code in R
Include R packages and import train data.
Import just 22M rows to decrease the time.
```r
library(tidyverse) # metapackage with lots of helpful functions
library(xgboost)
library(caret)
library(magrittr)
library(Matrix)
library(lightgbm)

tr <- read.csv("../input/train.csv",nrows=22000000) %>% select(-key)
```

Change datatime and deleted each features with anomaly value.

```r
tr  <- tr %>%
	mutate(pickup_datetime = as.POSIXct(pickup_datetime)) %>%
	mutate(hour = as.numeric(format(pickup_datetime, "%H"))) %>%
	mutate(min = as.numeric(format(pickup_datetime, "%M"))) %>%
	mutate(year = as.factor(format(pickup_datetime, "%Y"))) %>%
	mutate(day = as.factor(format(pickup_datetime, "%d"))) %>%
	mutate(month = as.factor(format(pickup_datetime, "%m"))) %>%
	mutate(Wday = as.factor(weekdays(pickup_datetime))) %>%
	mutate(hour_class = as.factor(ifelse(hour < 7, "Overnight",
		ifelse(hour < 11, "Morning",
		ifelse(hour < 16, "Noon",
		ifelse(hour < 20, "Evening",
		ifelse(hour < 23, "night", "overnight") ) ))))) %>%
	filter(fare_amount > 0 & fare_amount <= 500) %>%
	filter(pickup_longitude > -80 && pickup_longitude < -70) %>%
	filter(pickup_latitude > 35 && pickup_latitude < 45) %>%
	filter(dropoff_longitude > -80 && dropoff_longitude < -70) %>%
	filter(dropoff_latitude > 35 && dropoff_latitude < 45) %>%
	filter(passenger_count > 0 && passenger_count < 10) %>%
	mutate(pickup_latitude = (pickup_latitude * pi)/180) %>%
	mutate(dropoff_latitude = (dropoff_latitude * pi)/180) %>%
	mutate(dropoff_longitude = (dropoff_longitude * pi)/180) %>%
	mutate(pickup_longitude = (pickup_longitude * pi)/180 ) %>%
	mutate(dropoff_longitude = ifelse(is.na(dropoff_longitude) == TRUE, 0,dropoff_longitude)) %>%
	mutate(pickup_longitude = ifelse(is.na(pickup_longitude) == TRUE, 0,pickup_longitude)) %>%
	mutate(pickup_latitude = ifelse(is.na(pickup_latitude) == TRUE, 0,pickup_latitude)) %>%
	mutate(dropoff_latitude = ifelse(is.na(dropoff_latitude) == TRUE, 0,dropoff_latitude)) %>%
	select(-pickup_datetime,-hour_class,-min)
```

Calculate disatnce with haversine formula.
```r
tr$dlat <- tr$dropoff_latitude - tr$pickup_latitude
tr$dlon <- tr$dropoff_longitude - tr$pickup_longitude

#Compute haversine distance
tr$hav = sin(tr$dlat/2.0)**2+cos(tr$pickup_latitude)*cos(tr$dropoff_latitude) * sin(tr$dlon/2.0)**2
tr$haversine <- 2 * R_earth * asin(sqrt(tr$hav))

#Compute Bearing distance
tr$dlon <- tr$pickup_longitude - tr$dropoff_longitude
tr$bearing <- atan2(sin(tr$dlon * cos(tr$dropoff_latitude)), cos(tr$pickup_latitude) * sin(tr$dropoff_latitude) - sin(tr$pickup_latitude) * cos(tr$dropoff_latitude) * cos(tr$dlon))
```

Below is the function that calculate between pickup and dropoff with latitude and longitude
```r
sphere_dist <- function(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
{
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    #Compute  distance
    a = sin(dlat/2.0)**2 + cos(pickup_lat) * cos(dropoff_lat) * sin(dlon/2.0)**2

    return (2 * R_earth * asin(sqrt(a)))

}
```

Calculate the coordinate and distance of main stops with "sphere_dist" func.
```r
#Places Latitude and Longitude
jfk_coord_lat <- (40.639722 * pi)/180
jfk_coord_long <- (-73.778889 * pi)/180
ewr_coord_lat <- (40.6925 * pi)/180
ewr_coord_long <- (-74.168611 * pi)/180
lga_coord_lat <- (40.77725 * pi)/180
lga_coord_long <- (-73.872611 * pi)/180
liberty_statue_lat <- (40.6892 * pi)/180
liberty_statue_long <- (-74.0445 * pi)/180
nyc_lat <- (40.7141667 * pi)/180
nyc_long <- (-74.0063889 * pi)/180

tr$JFK_dist = sphere_dist(tr$pickup_latitude, tr$pickup_longitude, jfk_coord_lat, jfk_coord_long) + sphere_dist(jfk_coord_lat, jfk_coord_long, tr$dropoff_latitude, tr$dropoff_longitude)
tr$EWR_dist = sphere_dist(tr$pickup_latitude, tr$pickup_longitude, ewr_coord_lat, ewr_coord_long) +  sphere_dist(ewr_coord_lat, ewr_coord_long, tr$dropoff_latitude, tr$dropoff_longitude)
tr$lga_dist = sphere_dist(tr$pickup_latitude, tr$pickup_longitude, lga_coord_lat, lga_coord_long) + sphere_dist(lga_coord_lat, lga_coord_long, tr$dropoff_latitude, tr$dropoff_longitude)
tr$sol_dist = sphere_dist(tr$pickup_latitude, tr$pickup_longitude, liberty_statue_lat, liberty_statue_long) + sphere_dist(liberty_statue_lat, liberty_statue_long, tr$dropoff_latitude, tr$dropoff_longitude)
tr$nyc_dist = sphere_dist(tr$pickup_latitude, tr$pickup_longitude, nyc_lat, nyc_long) + sphere_dist(nyc_lat, nyc_long, tr$dropoff_latitude, tr$dropoff_longitude)
```

Separate training data into 2 sets: 1 for training and 1 for validation
```r
tri <- createDataPartition(target, p = 0.9, list = F) %>% c()
dtrain <- Matrix(as.matrix(tr[tri, ]),sparse=TRUE)
dval <- Matrix(as.matrix(tr[-tri, ]),sparse=TRUE)

categorical_feature <- c("day","month","year")

lgb.train = lgb.Dataset(data=dtrain,label=target[tri],categorical_feature =categorical_feature)
lgb.valid = lgb.Dataset(data=dval,label=target[-tri],categorical_feature =categorical_feature)
```

Train the data
```r
lgb.model <- lgb.train(
  params = list(objective = "regression"
  , metric = "rmse"
    ,num_boost_round=10000)
  , data = lgb.train
  , valids = list(val = lgb.valid)
  , boosting_type = "gbdt"
  , seed = 0
)
```
Then adjust same feature processing to test set like below.
```r
te <- read.csv("../input/test.csv",colClasses=c("key"="character","pickup_datetime"="POSIXct",
                  "dropoff_longitude"="numeric","pickup_longitude"="numeric","dropoff_latitude"="numeric","pickup_latitude"="numeric",
                  "passenger_count"="integer"),header=TRUE,sep=',') %>% select(-key)

te  <- te %>%
    mutate(pickup_datetime = as.POSIXct(pickup_datetime)) %>%
    mutate(hour = as.numeric(format(pickup_datetime, "%H"))) %>%
	mutate(min = as.numeric(format(pickup_datetime, "%M"))) %>%
	mutate(year = as.factor(format(pickup_datetime, "%Y"))) %>%
	mutate(day = as.factor(format(pickup_datetime, "%d"))) %>%
	mutate(month = as.factor(format(pickup_datetime, "%m"))) %>%
	mutate(Wday = as.factor(weekdays(pickup_datetime))) %>%
	mutate(hour_class = ifelse(hour < 7, "Overnight",
		ifelse(hour < 11, "Morning",
		ifelse(hour < 16, "Noon",
		ifelse(hour < 20, "Evening",
		ifelse(hour < 23, "night", "overnight") ) )))) %>%
	mutate(pickup_latitude = (pickup_latitude * pi)/180) %>%
	mutate(dropoff_latitude = (dropoff_latitude * pi)/180) %>%
	mutate(dropoff_longitude = (dropoff_longitude * pi)/180) %>%
	mutate(pickup_longitude = (pickup_longitude * pi)/180 ) %>%
	mutate(dropoff_longitude = ifelse(is.na(dropoff_longitude) == TRUE, 0,dropoff_longitude)) %>%
	mutate(pickup_longitude = ifelse(is.na(pickup_longitude) == TRUE, 0,pickup_longitude)) %>%
	mutate(pickup_latitude = ifelse(is.na(pickup_latitude) == TRUE, 0,pickup_latitude)) %>%
	mutate(dropoff_latitude = ifelse(is.na(dropoff_latitude) == TRUE, 0,dropoff_latitude)) %>%
	select(-pickup_datetime,-hour_class,-min)
te$dlat <- te$dropoff_latitude - te$pickup_latitude
te$dlon <- te$dropoff_longitude - te$pickup_longitude

#Compute haversine distance
te$hav = sin(te$dlat/2.0)**2 + cos(te$pickup_latitude) * cos(te$dropoff_latitude) * sin(te$dlon/2.0)**2
te$haversine <- 2 * R_earth * asin(sqrt(te$hav))


te$dlon <- te$pickup_longitude - te$dropoff_longitude
te$bearing = atan2(sin(te$dlon * cos(te$dropoff_latitude)), cos(te$pickup_latitude) * sin(te$dropoff_latitude) - sin(te$pickup_latitude) * cos(te$dropoff_latitude) * cos(te$dlon))


te$JFK_dist = sphere_dist(te$pickup_latitude, te$pickup_longitude, jfk_coord_lat, jfk_coord_long) + sphere_dist(jfk_coord_lat, jfk_coord_long, te$dropoff_latitude, te$dropoff_longitude)
te$EWR_dist = sphere_dist(te$pickup_latitude, te$pickup_longitude, ewr_coord_lat, ewr_coord_long) +  sphere_dist(ewr_coord_lat, ewr_coord_long, te$dropoff_latitude, te$dropoff_longitude)
te$lga_dist = sphere_dist(te$pickup_latitude, te$pickup_longitude, lga_coord_lat, lga_coord_long) + sphere_dist(lga_coord_lat, lga_coord_long, te$dropoff_latitude, te$dropoff_longitude)
te$sol_dist = sphere_dist(te$pickup_latitude, te$pickup_longitude, liberty_statue_lat, liberty_statue_long) + sphere_dist(liberty_statue_lat, liberty_statue_long, te$dropoff_latitude, te$dropoff_longitude)
te$nyc_dist = sphere_dist(te$pickup_latitude, te$pickup_longitude, nyc_lat, nyc_long) + sphere_dist(nyc_lat, nyc_long, te$dropoff_latitude, te$dropoff_longitude)

te <- te %>% select(-dlat,-dlon,-hav)

te$year <- as.numeric(te$year)
te$month <- as.numeric(te$month)
te$Wday <- as.numeric(te$Wday)
te$day <- as.numeric(as.factor(te$day))

dtest1 <- Matrix(as.matrix(te),sparse=TRUE)
```

And can have predicted fare after adjusting trained model to above test set.
```r
fare_amount = predict(lgb.model, dtest1)
```

This simple code makes quite good prediction. Will cover other ideas adpated to this competition in other post.
