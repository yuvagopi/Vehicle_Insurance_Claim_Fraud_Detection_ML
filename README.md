# Vehicle Insurance Claim Fraud Detection Using Machine Learning models

## Description
This project has been developed to predict the target variable using various independent variables with the help of machine learning models. 
With this project, the goal was to solve the problem of predicting the accuracy of a vehicle insurance claim. 
This was done during my summer internship at Henotics Technologies Pvt Ltd, Hyderabad.

## Abstract
The project is done on fraud detection of Vehicle insurance claims. The Vehicle insurance fraud involves conspiring to make false or exaggerated claims involving property damage or personal injuries following an accident. Some common examples include staged accidents where fraudsters deliberately “arrange” for accidents to occur; the use of phantom passengers where people who were not even at the scene of the accident claim to have suffered grievous injury, and make false personal injury claims where personal injuries are grossly exaggerated. While some insurance frauds are categorised as false claims, some frauds are classified as false policies. As a result, innocent people fall prey to fraud insurance providers and fraud insurance agents. Thus this project is to predict the trueness of the claim for vehicle insurance by providing whether it includes fraud or not ,by using popular AIML models.

# About Dataset
This data set have 15421 records of  23 columns(attributes/variables).
The target variable is “ Fraud_Found” .
And our target is to detect if a claim application is fraudulent or non-fraudulent.
The remaining variables are { Month, WeekOfMonth, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed, Gender, MaritalStatus, Age, Fault, PolicyType,  VehicleCategory, PolicyNumber, RepNumber, Deductible, DriverRating, PoliceReportFiled,WitnessPresent, AgentType, BasePolicy } (22 columns).

# Different ML models used
Different Models Used in this AIML Classification are:
  LogisticRegression() 
  DecisionTreeClassifier()
  RandomForestClassifier( ) 
  ExtraTreesClassifier() 
  KNeighborsClassifier() 
  SVC(probability=True) 
  BaggingClassifier()
  GradientBoostingClassifier()
  LGBMClassifier()
  XGBClassifier()
We used our train dataset to build the above models and used our test data to check the accuracy and performance of our models.
