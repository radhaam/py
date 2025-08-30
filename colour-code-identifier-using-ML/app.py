import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load dataset
csv_path = 'colors.csv'
index = ["color", "color_name", "hex", "R", "G", "B"]
data = pd.read_csv(csv_path, names=index, header=None) # columns: color_name,R,G,B

X = data[['R','G','B']]  # features
y = data['color_name']   # labels

# Step 2: Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Step 3: Predict a new color
test_color = [[130, 0, 200]]  # some purple-like color
prediction = model.predict(test_color)

print("Predicted Color:", prediction[0])
