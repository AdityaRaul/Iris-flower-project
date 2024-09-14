#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df=pd.read_csv("C:\\Users\\ADITYA\\Desktop\\iris flower project\\iris.csv",\
               names=['sl','sw','pl','pw','class'])
df['class']=df['class'].\
map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
df.sample(5)
X=df.drop(columns=['class'])
Y=df['class']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)
#################################################################################
import warnings
warnings.filterwarnings('ignore')
import tkinter.messagebox as m
def knn():
    global acc_knn
    from sklearn.neighbors import KNeighborsClassifier
    K=KNeighborsClassifier(n_neighbors=5)
    #train the model using 120 flower data
    K.fit(X_train,Y_train)
    #test the model using 30 flower input
    Y_pred_knn=K.predict(X_test)
    
    #find accuracy of knn model
    from sklearn.metrics import accuracy_score
    acc_knn=accuracy_score(Y_test,Y_pred_knn)
    acc_knn=round(acc_knn*100,2)
    m.showinfo(title="KNN",message="accuracy  is" +str(acc_knn)+"%")
def lg():
    global acc_lg
    from sklearn.linear_model import LogisticRegression
    L=LogisticRegression()
    L.fit(X_train,Y_train)
    Y_pred_lg=L.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_lg=accuracy_score(Y_test,Y_pred_lg)
    acc_lg=round(acc_lg*100,2)
    m.showinfo(title="LG",message="accuracy  is" +str(acc_lg)+"%")
def dt():
    global acc_dt
    from sklearn.tree  import DecisionTreeClassifier
    D=DecisionTreeClassifier()
    D.fit(X_train,Y_train)
    Y_pred_dt=D.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_dt=accuracy_score(Y_test,Y_pred_dt)
    acc_dt=round(acc_dt*100,2)
    m.showinfo(title="DT",message="accuracy  is" + str(acc_dt)+"%")
def nb():
    global acc_nb
    from sklearn.naive_bayes  import GaussianNB
    N=GaussianNB()
    # train the model
    N.fit(X_train,Y_train)
    # test
    Y_pred_nb=N.predict(X_test)
    # find accuracy
    from sklearn.metrics import accuracy_score
    acc_nb=accuracy_score(Y_test,Y_pred_nb)
    acc_nb=round(acc_nb*100,2)
    m.showinfo(title="NB",message="accuracy  is" + str(acc_nb)+"%")
###############################################################################
def compare():
    import matplotlib.pyplot as plt
    models=['KNN','LG','DT','NB']
    accuracy=[acc_knn,acc_lg,acc_dt,acc_nb]
    plt.bar(models,accuracy,color=['yellow','green','red','blue'])
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()
def Submit():
    sl = float(Esl.get())
    sw = float(Esw.get())
    pl = float(Epl.get())
    pw = float(Epw.get())

    # Use the Logistic Regression model for prediction
    result = L.predict([[sl, sw, pl, pw]])
    if result[0] == 0:
        flower = "SETOSA"
    elif result[0] == 1:
        flower = "VERSICOLOR"
    else:
        flower = "VIRGINICA"
        
    m.showinfo(title='IRIS',message=flower)
def Reset():
    Esl.delete(0,END)
    Esw.delete(0,END)
    Epl.delete(0,END)
    Epw.delete(0,END)
    
from tkinter import*
w=Tk()
w.title("IRIS PROJECT")
w.configure(background='purple')
#1st row
L1=Label(w,text="IRIS FLOWER PROJECT",font=('arial',15,'bold'),bg='yellow')
L1.grid(row=1,column=1,columnspan=4)
#2nd row
Bknn=Button(w,text="KNN",font=('arial',15,'bold'),command=knn,bg='red',fg='white')
Blg=Button(w,text="LG",font=('arial',15,'bold'),command=lg,bg='red',fg='white')
Bdt=Button(w,text="DT",font=('arial',15,'bold'),command=dt,bg='red',fg='white')
Bnb=Button(w,text="NB",font=('arial',15,'bold'),command=nb,bg='red',fg='white')
Bcmp=Button(w,text="Compare",font=('arial',15,'bold'),command=compare,bg='red',fg='white')
Bknn.grid(row=2,column=1)
Blg.grid(row=2,column=2)
Bdt.grid(row=2,column=3)
Bnb.grid(row=2,column=4)
Bcmp.grid(row=3,column=2,columnspan=2)
#4th row
L2=Label(w,text="PREDICT FOR A NEW FLOWER",font=('arial',15,'bold'),bg='yellow')
L2.grid(row=4,column=1,columnspan=4)
#row 5
Lsl=Label(w,text="SL",font=('arial',15,'bold'),bg='green',fg='white')
Esl=Entry(w,font=('arial',15,'bold'),width=5,bg='aqua')
Lsw=Label(w,text="SW",font=('arial',15,'bold'),bg='green',fg='white')
Esw=Entry(w,font=('arial',15,'bold'),width=5,bg='aqua')
Lsl.grid(row=5,column=1)
Esl.grid(row=5,column=2)
Lsw.grid(row=5,column=3)
Esw.grid(row=5,column=4)
#row 6
Lpl=Label(w,text="PL",font=('arial',15,'bold'),bg='green',fg='white')
Epl=Entry(w,font=('arial',15,'bold'),width=5,bg='aqua')
Lpw=Label(w,text="PW",font=('arial',15,'bold'),bg='green',fg='white')
Epw=Entry(w,font=('arial',15,'bold'),width=5,bg='aqua')
Lpl.grid(row=6,column=1)
Epl.grid(row=6,column=2)
Lpw.grid(row=6,column=3)
Epw.grid(row=6,column=4)
#7th row
Bsub=Button(w,text="Submit",font=('arial',15,'bold'),command=Submit,bg='black',fg='white')
Bres=Button(w,text="Reset",font=('arial',15,'bold'),command=Reset,bg='black',fg='white')
Bsub.grid(row=7,column=1,columnspan=2)
Bres.grid(row=7,column=3,columnspan=2)
w.mainloop()

