import streamlit as st
from streamlit_option_menu import option_menu 
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns 
import matplotlib.pyplot as plt
import sys

dt=pd.read_csv("C:/Users/AlfredRomarioG/OneDrive - Cittabase Solutions Private Limited/Desktop/Project/train.csv")
df=pd.read_csv("C:/Users/AlfredRomarioG/OneDrive - Cittabase Solutions Private Limited/Desktop/Project/train.csv")
#Sidebar

#drop duplicates
# dt=dt.drop_duplicates()
df=df.drop_duplicates()
df.duplicated().sum()

#Feature Engineering
#day
Category_A = []

for i in df['day']:
      if i <= 10:
             Category_A.append(1)
      elif i <= 20:
             Category_A.append(2)
      else:
             Category_A.append(3)
  
df['day_cat'] = Category_A
# df = df.drop(columns=['Cat_A_day'])

#Dur
Category_1=[]

for i in df['dur']:
        if i<1000:
              Category_1.append(1)
        elif i>=1000 and i<2000:
              Category_1.append(2)
        elif i>=2000 and i<3000:
              Category_1.append(3)
        elif i>=3000 and i<4000:
              Category_1.append(4)
        else:
              Category_1.append(5)

df['dur_cat']=Category_1


Category_B=[]
for i in df['num_calls']:
        if i<10:
              Category_B.append(1)
        elif i>=10 and i<20:
              Category_B.append(2)
        elif i>=20 and i<30:
              Category_B.append(3)
        elif i>=30 and i<40:
              Category_B.append(4)
        elif i>=40 and i<50:
              Category_B.append(5)
        else:
              Category_B.append(6)

df['num_calls_cat']=Category_B

#Categorizing Age

Category_C=[]

for i in df['age']:
      if i <20:
            Category_C.append(1)
      elif i>=20 and i<40:
            Category_C.append(2)
      elif i>=40 and i<60:
            Category_C.append(3)
      elif i>=60 and i<80:
            Category_C.append(4)
      else:
            Category_C.append(5)

df['age_cat']=Category_C

df.info()
df['target']=df['y'].map({"yes":1,"no":0})


#Outlier Deduction
# sns.set(style="whitegrid")
# sns.boxplot(x=df['age'], color='Chartreuse')
# plt.show()






#####################STREAMLIT SCRIPT###########################


with st.sidebar:
    selected=option_menu(
        menu_title="Menu",
        options=["Home","About the dataset","Data Preprocessing","Analysis"],
        icons=["house","database","sort-up","graph-up-arrow"],
        menu_icon="cast",
        default_index=0,

    )

if selected=="Home":
    
    st.title("Customer Conversion Prediction")
    st.header("*:red[Customer Conversion]*")
    st.text("Customer conversion refers to the process of turning a prospect into a paying customer.\n"
            "An example in the insurance field is to identify the customers that are most likely \n"
            "to convert beforehand so that they can be specifically targeted via call.")
    
if selected=="About the dataset":
    st.header(':red[Varaible Description]')
    st.markdown(
    """
    **Features:**
    - age(numeric) : age of the people
    - job : type of job
    - marital : marital status
    - educational_qual : education status
    - call_type : contact communication type
    - day: last contact day of the month (numeric)
    - mon: last contact month of year
    - dur: last contact duration, in seconds (numeric)
    - num_calls: number of contacts performed during this campaign and for this client
    - prev_outcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
    """
    )
    st.markdown(
    """
    **Output variable (desired target):**
    - y - has the client subscribed to the insurance?
    """       
    )
    st.header(":red[Dataset]")
    st.markdown("*displaying the first 100 rows*")
    st.write(dt.head(100))
    st.subheader(":red[Basic Statistics]")
    st.write(dt.describe())
    st.markdown('### :red[DataTypes]')
    dat=pd.DataFrame(dt.dtypes)
    st.write(dat)


if selected=="Data Preprocessing":
      clicked=option_menu(
        menu_title="Data Preprocessing",
        options=["Data Cleaning","Outlier Deduction","Feature Engineering"],
        icons=["search","dash-circle","diagram-3"],
        menu_icon="sort-up",
        default_index=0,
        orientation="horizontal"
      )
      if clicked=="Data Cleaning":
            st.header(":red[Removing Duplicates]")
            st.write("**Duplicate count:** ",dt.duplicated().sum())
            st.subheader(":green[After removing]")
            st.write("**Duplicate count:** ",df.duplicated().sum())
            st.header(":red[Checking for Nulls]")
            st.table(dt.isnull().sum())
      if clicked=="Outlier Deduction":
            st.header(':red[Outlier]')
            st.text("An outlier is a data point that differs significantly from other observations.\n"
                    "An outlier may be due to a variability in the measurement,an indication of novel\n"
                    "data or it may be the result of experimental error the latter are sometimes excluded\n"
                    "from the data set.")
            st.subheader(":red[Inter Quartile Range]")
            st.text("The interquartile range defines the difference between the third and the first quartile.\n"
                    "Quartiles are the partitioned values that divide the whole series into 4 equal parts.\n"
                    "So,there are 3 quartiles. First Quartile is denoted by Q1 known as the lower quartile,\n"
                    "the second Quartile is denoted by Q2 and the third Quartile is denoted by Q3 known as\n"
                    "the upper quartile.Therefore, the interquartile range is equal to the upper quartile\n"
                    "minus lower quartile.")
            st.markdown("**Interquartile range = Upper Quartile – Lower Quartile = Q­3 – Q­1**")
            
            okay=option_menu(
            menu_title="Remove Outliers for",
            options=["Age","Day"],
            icons=["award","calendar3"],
            menu_icon="dash-circle",
            #default_index=0,
            )
            if okay=="Age":
                  st.header(":red[Outlier for Age]")
                  fig, ax=plt.subplots()
                  sns.boxplot(x='age',data=dt,ax=ax)
                  st.pyplot(fig)
                  #detecting Outlier for Age column
                  q1,q3=np.percentile(df["age"],[25,75])
                  IQR=q3-q1
                  upper=q3+1.5*IQR
                  lower=q1-1.5*IQR
                  st.write("Upper age bound:",upper)
                  st.write("Lower age bound :", lower)
                  df.age = df.age.clip(10.5,70.5)
                  st.write("## :green[After removal of outlier]")
                  fig, ax=plt.subplots()
                  sns.boxplot(x='age',data=df,ax=ax,)
                  st.pyplot(fig)

            if okay=="Day":
                  st.header(":red[Outlier for Day]")
                  fig, ax=plt.subplots()
                  sns.boxplot(x='day',data=dt,ax=ax)
                  st.pyplot(fig)
                  #detecting Outlier for Age column
                  q1,q3=np.percentile(df["day"],[25,75])
                  IQR=q3-q1
                  upper=q3+1.5*IQR
                  lower=q1-1.5*IQR
                  st.write("Upper Day bound:",upper)
                  st.write("Lower Day bound :", lower)
                  st.subheader('No outlier found')

      if clicked=="Feature Engineering":
            st.header(":red[Categorizing Features to create new Features]")
            st.subheader(":green[Age]")
            st.write(df.age_cat.value_counts())
            st.subheader(":green[Day]")
            st.write(df.day_cat.value_counts())
            st.subheader(":green[Num Calls]")
            st.write(df.num_calls_cat.value_counts())
            st.subheader(":green[Duration]")
            st.write(df.dur_cat.value_counts())
            
if selected=="Analysis":
    st.write("No.of people successfully converted:", df['y'].value_counts()['yes'])
    st.write("No.of people failed to convert:", df['y'].value_counts()['no'])

    


      
    
