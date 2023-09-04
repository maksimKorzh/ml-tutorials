# Packages
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data from CSV file
data = pd.read_csv('cmk_videos.csv')

# Get programming language
def get_lang(row):
  i = 0
  for lang in [
    r'\bpython\b',
    r'\bjavascript\b',
    r'\bjs\b',
    r'c\+\+',
    r'\bc\b',
    r'\bgolang\b',
    r'\bgo\b',
    r'\bassembly\b'
  ]:
    i += 1
    if re.search(lang, row['title'].lower()):
      if 'js' in lang: return 2
      elif 'golang' in lang: return 6
      else: return i
  return 9

# Get video topic
def get_topic(row):
  i = 0
  for topic in [
    r'scrap',
    r'\bbeautifulsoup\b',
    r'\bflask\b',
    r'\bweb\b',
    r'\braycasting\b',
    r'\blinux\b',
    r'\btext editor\b',
    r'\bkim\b',
    r'\barduino\b',
    r'\bopencv\b',
    r'conver',
    r'6502'
  ]:
    i += 1
    if re.search(topic, row['title'].lower()):
      if 'soup' in topic: return 1
      elif 'flask' in topic: return 4
      else: return i
  return 13

# Classify misc keywords
def get_misc(row):
  if 'how to' in row['title'].lower(): return 1
  elif 'live coding' in row['title'].lower(): return 2
  elif 'scrap' in row['title'].lower(): return 3
  else: return 4
  
# Convert Y/M/W to days
def get_days(row):
  days = int(row['published'].split(' ')[0])
  if 'year' in row['published']: return days * 365
  elif 'month' in row['published']: return days * 30
  elif 'week' in row['published']: return days * 7
  else: return days

# Categorize views
def is_worth(row):
  if row['views'] < 1000: return 0
  else: return 1

# Feature engineering
data['lang'] = data.apply(get_lang, axis=1)
data['topic'] = data.apply(get_topic, axis=1)
data['misc'] = data.apply(get_misc, axis=1)
data['days'] = data.apply(get_days, axis=1)
data['worth'] = data.apply(is_worth, axis=1)

# FIlter dataset
data = data[['lang', 'topic', 'misc', 'days', 'worth']]

# Split data into input features (X) and target variable (y)
X = data.drop('worth', axis=1)
y = data['worth']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test).astype('int')

# Main driver
if __name__ == '__main__':
  # Measure model accuracy
  print('===================================')
  print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
  print('===================================')
  feature_importances = classifier.feature_importances_
  importance = dict(zip(X.columns, feature_importances))
  sort_importance = {key: val for key, val in sorted(importance.items(), key=lambda x: x[1])}
  for feature, importance in sort_importance.items():
    print(f'{feature}: {importance}')
  print('===================================')
  
  # Save model
  joblib.dump(classifier, 'classifier.pkl')
  print('Model is saved as "classifier.pkl"')
  print('===================================')
