# Song-Genre-Prediction
Predicting the genre of songs for given song ids. User song listening activity history and song duration related data is provided for learning.

1.	Provided Problem Statement
Predict the genre of some tracks with following available data:
session.csv: This file contains records of various user session listening specific tracks of their choice at different point of time. More specifically each record has set of three field user_id, song_id and timestamp.   
 
Execution Environment & Required Library:

1) Python 2.x
2) Libraries
   - Scikit Learn
   - Numpy
   - Pandas
   - Matplotlib
   
tracks.csv: It has records of different tracks with their time duration and genre. Each record consists of three field song_id, duration and genre respectively.
 
tracks_to_complete.csv: This file contains test data having song_id for which genre needs to be predicted.
