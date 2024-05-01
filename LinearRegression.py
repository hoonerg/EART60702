import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

training_data = np.load('./processed/009/training.npy')
validation_data = np.load('./processed/009/validation.npy')
test_data_1 = np.load('./processed/009/test_1.npy')
test_data_2 = np.load('./processed/009/test_2.npy')
test_data_3 = np.load('./processed/009/test_3.npy')

X_train = training_data[:, :528]
y_train = training_data[:, 545]
X_valid = validation_data[:, :528]
y_valid = validation_data[:, 545]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_valid = model.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Validation RMSE: {rmse_valid}")

X_test_1 = test_data_1[:, :528]
y_test_1 = test_data_1[:, 545]
X_test_2 = test_data_2[:, :528]
y_test_2 = test_data_2[:, 545]
X_test_3 = test_data_3[:, :528]
y_test_3 = test_data_3[:, 545]

y_pred_test_1 = model.predict(X_test_1)
rmse_test_1 = np.sqrt(mean_squared_error(y_test_1, y_pred_test_1))
print(f"Test 1 RMSE: {rmse_test_1}")

y_pred_test_2 = model.predict(X_test_2)
rmse_test_2 = np.sqrt(mean_squared_error(y_test_2, y_pred_test_2))
print(f"Test 2 RMSE: {rmse_test_2}")

y_pred_test_3 = model.predict(X_test_3)
rmse_test_3 = np.sqrt(mean_squared_error(y_test_3, y_pred_test_3))
print(f"Test 3 RMSE: {rmse_test_3}")
