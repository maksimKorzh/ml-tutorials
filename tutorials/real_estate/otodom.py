# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import requests

# Download dataset
print('Downloading dataset... ', end='')
response = requests.get('https://github.com/maksimKorzh/ml-tutorials/releases/download/0.1/otodom.csv')
with open('otodom.csv', 'wb') as f: f.write(response.content)
print('Done!')

# Allow working with Data Frames copies
pd.set_option('mode.chained_assignment', None)

# Load data from CSV
data = pd.read_csv('otodom.csv')

# Extract features from characteristins
def preprocess_data(data, feature, split_str):
  # Extract feature
  data[feature] = data['characteristics'].apply(
    lambda x:
      x.split(split_str)[-1]
       .split("'")[0]
       .replace('garret', '1')
       .replace('cellar', '0')
       .replace('higher_', '')
       .replace('ground_floor', '1')
       .replace('floor_', '')
       .replace('more', '5')
       .replace('[{', '0')
  )

  # Drop N/A values
  data = data[data[feature] != '']
  try: data[feature] = data[feature].astype(float)
  except:
    onehot = pd.get_dummies(data[feature])
    try: onehot.drop(['0'], axis=1, inplace=True)
    except: pass
    data = data.join(onehot)
  return data

# Extract features
data = preprocess_data(data, 'price', "'price': '")
data = preprocess_data(data, 'floor', "'floor_no': '")
data = preprocess_data(data, 'area', "'m': '")
data = preprocess_data(data, 'rooms', "'rooms_num': '")
data = preprocess_data(data, 'year', "'build_year': '")
data = preprocess_data(data, 'heating', "'heating': '")
data = preprocess_data(data, 'construction_status', "'construction_status': '")
data = preprocess_data(data, 'building_ownership', "'building_ownership': '")

# Drop redundant features
data.drop([
  'id',
  'advertiserType',
  'advertType',
  'createdAt',
  'modifiedAt',
  'description',
  'features',
  'characteristics',
  'street_name',
  'street_number',
  'subdistrict',
  'district',
  'city',
  'county',
  'province',
  'postalCode',
  'map_url',
  'title',
  'heating',
  'construction_status',
  'building_ownership'
], axis=1, inplace=True)

# Print eventual dataset
print(data.info())

# Split data into source features (X) and target variable (y)
X = data.drop(['price'], axis=1)
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = RandomForestRegressor(n_estimators=40, random_state=23)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
ACC = model.score(X_train, y_train)
acc = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)

# Print results
print('===================================')
print(' Results:')
print('===================================')
print('ACC:', ACC)
print('Acc:', acc)
print('Avg:', y_test.mean())
print('MAE:', mae)
print('===================================')
print(' Feature importance:')
print('===================================')
feature_importances = model.feature_importances_
importance = dict(zip(X.columns, feature_importances))
sort_importance = {
  key: val for key, val in sorted(importance.items(),
  key=lambda x: x[1])
}
for feature, importance in sort_importance.items():
  print(f'{feature}: {importance}')
print('===================================')
