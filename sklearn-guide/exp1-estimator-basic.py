from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)
print(f"{clf}")

X = [[1, 2, 3],
     [11, 12, 13]]
y = [0, 1]

clf.fit(X, y)

print(f"{clf.predict(X)}")
print(f"{clf.predict([[1, 2, 3], [14, 15, 16]])}")
