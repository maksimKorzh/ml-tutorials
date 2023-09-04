# Packages
import joblib
import pandas as pd
from train import get_lang, get_topic, get_misc

# Load a pre-trained model
model = joblib.load('classifier.pkl')

# Predict result
def predict_result(title):
  # Loop over the range of days
  for days in range(1, 1460):
     # Create dataset
     data = pd.DataFrame.from_dict({'title': [title]})

     # Encode features
     data['lang'] = data.apply(get_lang, axis=1)
     data['topic'] = data.apply(get_topic, axis=1)
     data['misc'] = data.apply(get_misc, axis=1)
     data['days'] = [days]

     # Drop title
     data = data.drop('title', axis=1)

     # Predict results
     y_pred = model.predict(data)

     # Return success
     if y_pred[0] == 1: return days

  # Return failure
  return 0

# Main driver
while True:
  # Get user input
  print('Enter your video title:')
  title = input('>')
  if title == 'quit': break
  days = predict_result(title)
  if days > 0: print('Your video is about to gain 1000 views in', days)
  else: print('Your video sucks, you should try something else')
