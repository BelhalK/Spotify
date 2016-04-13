import pandas as pd
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt

# Load D (data set), X (exogeous predictors), Y (endogenous response).
import pandas as pd
print 'Loading the data'
df1 = pd.read_csv("data_sample/end_song_sample.csv", delimiter=',', low_memory = False)
df2 = pd.read_csv("data_sample/user_data_sample.csv", delimiter=',', low_memory = False)
#Adding a '1' in front of each row
df1['count'] = pd.Series(int(1), index=df1.index)
#Counting the sum and the count of time spent and number tracks
df1_count = df1.groupby('user_id')
#Converting the dataframegroupby object into DataFrame
import numpy as np
df1_count = df1_count.aggregate(np.sum)
#Keeping only the two relevant metrics
df2_right = df1_count.loc[:,['ms_played', 'count']]
df2_left = df2.set_index('user_id')
df2_concat = pd.concat([df2_left, df2_right], axis=1, join='inner')

D = df2_concat[df2_concat['gender'] == 'female']


predictors = ['gender','country', 'acct_age_weeks',
'ms_played']
response = ['count']

# Extract into numpy arrays for numerical analysis.
X = np.asarray(D[predictors])
Y = np.asarray(D[response])


# Histogram and boxplot murder rate
fig = plt.figure()
# fig.suptitle('Histogram and Boxplot of Murder Rates', size=16)
gs = gridspec.GridSpec(2, 1, height_ratios = [5,1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

ax1.hist(Y, normed=False, color="#6495ED", bins=np.arange(0,14,0.75),
        alpha=.5)
ax1.set_ylabel('Number of people', fontweight = 'bold')
ax1.set_ylim([0, ax1.get_ylim()[1] + 1])
ax1.set_xticklabels([])

ax2.boxplot(Y,0,'rs',0,)
ax2.set_xlabel('Number of tracks listened', fontweight='bold')
ax2.set_yticklabels([])

# Extract five summary stats, and show on boxplot.
low = np.min(Y)
iqr = (np.around(np.percentile(Y,25), decimals=1),
    np.around(np.percentile(Y,75), decimals=1))
median = np.median(Y)
high = np.max(Y)

ax2.text(low -.2, 1.2, low)
ax2.text(iqr[0] -.2, 1.2, iqr[0])
ax2.text(median -.2, 1.2, median)
ax2.text(iqr[1] -.2, 1.2, iqr[1])
ax2.text(high -.2, 1.2, high)

plt.show()