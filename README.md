# customizable-regression
DataSet i.e. newdata.xlsx is a collection of the readings of various parameters taken over different periods from a LF (Ladle Furnace) and in our case, TEMP_DIFF is the target variable.

Customizable regression is a simple menu-driven python script where the user can select the regression technique, feature selection technique, test size, etc (from the available options) to perform regression over a given data set with minimal changes in the code and see the accuracy score, MAE, MSE, RMSE, etc. 

For now, I have commented out the section that performs feature scaling because that had very little impact on this particular dataset. However, if one prefers doing it (on some other dataset), the triple-quotes can be removed and it's good to go.

Duration from successive timestamps will be calculated and added in the coming days, along with some more regression and feature selection techniques, which I plan to learn in the coming days. 

Note: NEW_TEMP has been taken as the difference from the last reading of temperature.
