from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
print(f"{scaler}")

X = [[0, 15],
     [1, -10]]

x_scaler = scaler.fit(X).transform(X)

print(f"{x_scaler}")
print(f"{scaler.inverse_transform(x_scaler)}")
