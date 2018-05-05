
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#%matplotlib inline 


# In[2]:

#Reading the files
streams = pd.read_csv("../data/sessions.csv")
tracks = pd.read_csv("../data/tracks.csv")
unknown_tracks = pd.read_csv("../data/tracks_to_complete.csv")


# In[3]:

print("Size of streams is {}".format(streams.shape))
print("Size of tracks is {}".format(tracks.shape))
print("Size of unknown_track is {}".format(unknown_tracks.shape))


# In[4]:

#Function to split the date into useful components
def splitDatetime(data) :

   datatime = pd.DatetimeIndex(data.date)
   data['year'] = datatime.year
   data['month'] = datatime.month
   data['day'] = datatime.day
   data['hour'] = datatime.hour
   data['weekday'] = datatime.weekday
   return data


# In[5]:

#Converting the numerical timestamp to date format
streams['date'] = pd.to_datetime(streams['timestamp'],unit='s')

#Creating some more feature and deleting not needed one
streams = splitDatetime(streams)

del streams['timestamp']
del tracks['duration']


# In[8]:

print tracks['genre'].value_counts()


# In[9]:

#Some basic check to see songs_id presence in different files
print len(set(streams['song_id']))
print len(set(tracks['song_id']))
print len(set(unknown_tracks['song_id']))

print len(set(unknown_tracks['song_id']) & set(tracks['song_id']))
print len(set(unknown_tracks['song_id']) & set(streams['song_id']))

print len(set(streams['song_id']) - set(tracks['song_id']))


# There are 1265 songs for which we don't know the track out of which 1209 tracks can be mapped with stream data and remaining 56 we don't have clue. Matched 1209 tracks we need to map with the user_id to and their favourite genre

# In[10]:

#Merging streams and tracks to proceed further
streams_genre = streams.merge(tracks,how = 'left',on='song_id')


# In[11]:

#Verifying the Combined file with earlier one
print streams.shape
print streams_genre.shape

print streams.columns
print streams_genre.columns


# In[12]:

streams_genre = streams_genre.fillna(-1)


# In[13]:

#Creating additional column which can represent the person time and mood of listning songs
bins = [-1,5,11,17,20,24]
group_names = ['LateNight','Morning','Noon','Evening','Night']
streams_genre['daypart'] = pd.cut(streams_genre['hour'],bins,labels=group_names)


# In[14]:

#Converting Datatype Removing date column as important information has extracted out and visualising the dataset
streams_genre['user_id'] = streams_genre['user_id'].astype('category')
streams_genre['song_id'] = streams_genre['song_id'].astype('category')
streams_genre['year'] = streams_genre['year'].astype('int')
streams_genre['month'] = streams_genre['month'].astype('int')
streams_genre['day'] = streams_genre['day'].astype('int')
streams_genre['hour'] = streams_genre['hour'].astype('int')
streams_genre['weekday'] = streams_genre['weekday'].astype('int')
del streams_genre['date']
print streams_genre.head()


# In[15]:

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[16]:

streams_genre = MultiColumnLabelEncoder(columns = ['daypart']).fit_transform(streams_genre)


# In[17]:

#Splitting train and test set based on genre availability
train = streams_genre[streams_genre['genre'] != -1]
test = streams_genre[streams_genre['genre'] == -1]


# In[18]:

print train.shape
print test.shape
print (train.shape[0] + test.shape[0]) == streams_genre.shape[0] #Verfying the Results


# In[19]:

#Converting genre datatype to object and then coding them in numerical as sckit-learn expect the prediction classes in numeric
train['genre'] = train['genre'].astype('category')
print train['genre'].cat.categories
train['genre'] = train['genre'].cat.codes


# In[20]:

#Extracting the genre out as this is our label to predict 
Y = train.genre
del train['genre']
del test['genre']


# In[21]:

x_train, x_test, y_train, y_test = train_test_split(train,Y,test_size = .3,random_state = 42)


# In[22]:

rfc = RandomForestClassifier(n_estimators=150,random_state=42) #Taking Random forest classifier


# In[24]:

scores = cross_val_score(rfc, train, Y, cv=5) #Checking with cross validation score
print ("Cross Validation scores is:{}".format(scores))


# In[27]:

#Lets fit the model and check with train test split
rfc.fit(x_train,y_train)  
tree_pred = rfc.predict(x_test)


# In[28]:

print ("Accuracy Score on Validation set is:{}".format(accuracy_score(y_test,tree_pred)))


# In[29]:

#Getting and Plotting Feature importance

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
columns = train.columns
# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))


# In[30]:

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), columns[indices])
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[31]:

#As it's giving around 95% to 96% accuracy with mutiple split and cv score we can believe that our model is not overfit
#Now Lets train the model with full data and get the prediction
rfc.fit(train,Y)
pred = rfc.predict(test)


# In[45]:

#Making predicted dataframe
stream_predicted = pd.DataFrame({'song_id':test['song_id'],'pred':pred})


# In[46]:

#print stream_predicted[stream_predicted['song_id'] == 8589934592]


# In[47]:

song_by = stream_predicted.groupby(['song_id','pred']).pred.value_counts().to_frame()


# In[48]:

#song_by.head(12)


# In[49]:

song_by = song_by.groupby('song_id').idxmax()
final_df = pd.DataFrame(list(song_by.pred))


# In[50]:

print final_df.columns
del final_df[2]


# In[51]:

final_df.columns = ['song_id','predicted_genre']
print final_df['predicted_genre'].value_counts()


# In[52]:

#As Remaining songs we don't have information in session file we will assign them 'rock' as an genre as it's the most occuring genre and half of the tracks are from this genre
final_df = final_df.merge(unknown_tracks, how = 'right')


# In[53]:

final_df = final_df.fillna(4) #4 is code for rock genre


# In[54]:

#Now converting back to their Original value
#u'blues', u'electro', u'rap', u'reggae', u'rock'
final_df['predicted_genre'] = final_df['predicted_genre'].map({0.0:'blues',1.0:'electro',2.0:'rap',3.0:'reggae',4.0:'rock'})
print ("Final Predicted genre in Table count")
print final_df['predicted_genre'].value_counts()


# In[55]:

final_df.to_csv('solution.csv',index = False)


# In[ ]:



