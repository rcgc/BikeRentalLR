from gd import *

# https://www.kaggle.com/kratos2597/boom-bikes-linear-regression
df = pd.read_csv('registers.csv')

# Getting features
df_X = df['atemp']
df_Y = df['cnt']

# Feeling temperature in Celsius
df_X_train = df_X[:584]     # 000 - 583
df_X_test = df_X[584:]      # 584 - 729

# Total Rental of Bikes
df_Y_train = df_Y[:584]     # 000 - 583
df_Y_test = df_Y[584:]      # 584 - 729

# Plotting original training datasets
# plot_dataset(df_X_train, df_Y_train, 0)

# Scaling training datasets
df_X_train_scaled = (df_X_train - df_X_train.mean(axis=0)) / df_X_train.std()
df_X_test_scaled = (df_X_test - df_X_test.mean(axis=0)) / df_X_test.std()
# df_Y_train_scaled = (df_Y_train - df_Y_train.mean(axis=0)) / df_Y_train.std()

# Plotting scaled training datasets
plot_dataset(df_X_train_scaled, df_Y_train, 1)

# Start doing gd and epochs
bias, theta, gd_iterations_df = start(df_X_train_scaled, df_Y_train, 0.0005, 2000)

mean = np.mean(df_Y)

# Coefficient of regression in training
mse_train, R_train = coefficient_of_regression(df_X_train_scaled, df_Y_train, bias, theta, mean)
print("MSE in training: ", mse_train/730)
print("Coefficient of regression in training: ", R_train)

# Coefficient of regression in testing
mse_test, R_test = coefficient_of_regression(df_X_test_scaled, df_Y_test, bias, theta, mean)
print("MSE in testing: ", mse_test/730)
print("Coefficient of regression in testing: ", R_test)

# Plotting Linear Regression
plot_linear_regression(bias, theta, 1)

# Making predictions
make_predictions(bias, theta, 1, df_X_train.mean(axis=0), df_X_train.std())
#  5.8965  -> 1416
# 13.22605 -> 2947
# 26.8     -> 4553
# 31.345   -> 4785

plt.show()
