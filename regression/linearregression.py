import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

weeksalesdata = {'week': [1,2,3,4,5],
                 'sales': [1.2,1.8,2.6,3.2,3.8] }  

df = pd.DataFrame(weeksalesdata,columns=['week','sales']) 

plt.scatter(df['week'], df['sales'], color='green')
plt.title("Week and Sales in Thousands")
plt.xlabel("X - Weeks")
plt.ylabel("Y -Sales")
#plt.show()

X = df[['week']]
y = df['sales']
print( " type of y:", type(y))
print("\n")

regr = linear_model.LinearRegression()
regr.fit(X,y)

print("Intercept :\n",regr.intercept_)
print("\n")
print("Coefficients :\n",regr.coef_)

print("The regression equation is ", regr.coef_ , "* xi +", regr.intercept_)
print("\n")

for x in range(7,11):
    #weekx = input("Enter the Week Number :")
    weekx  = x 
    print("--------- Computing for Week :", weekx, "----------------")
    calcSales = (regr.coef_[0] *  int(weekx)) + regr.intercept_
    weekx = print(" Expected Sales in Thousands", weekx, " th week = ", calcSales)   
    print("\n")


pred = regr.predict(X) 
print(" pred :",pred)
plt.plot(X,pred)    

print("Adjusted R Squared for Regression model:",regr.score(X,y))

plt.show()