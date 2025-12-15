import pandas as pd
import numpy as np
# x=pd.Series(np.random.randint(10,60,10))
# print(x)
# print(x.head())
# print(x.tail())
# print(x.shape)
# print(x.size)

df1 = pd.DataFrame({
    'Name': ['Alice peter','Bob','Charlie george',None,'David john','Ava'],
    'Age': [25, 30, 35, 40, 28 , None],
    'City': ['NY','LA','SF',None ,'LA','NYC'],
    'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04',
        '2025-01-05', '2025-01-06']
})
# # print(df)
# print(df.head(2))
# print(df.tail(2))
# print(df.describe())
# print(df.info())
# print(df.shape)

# print(df.loc[:,'Name'])
# print(df.iloc[0,1:3])
# df['age']=['a','b','c','d','e']
# # print(df)

# print(df.drop(index=3,columns='age'))
# print(df)
# print(df.isna())
# print(df.dropna())
# print(df.fillna("A"))
# print(df.replace("LA","B"))

# print(df.sort_values('Age',ascending=False))
# print(df.sort_values())
# print(df.sort_index())
# print(df.rank(axis=0))

# print(df[df['Age']>20])
# print((df['Age']>20 ) & (df['City']=="NY"))
# print(df.query("Age>30 and Age<50"))

# print(df.groupby('City')['Age'].mean())
# print(df.groupby('City')['Age'].agg('mean'))

# print(df.groupby('City')['Age'].agg(['mean', 'max', 'min', 'count']))

# print(df.groupby('City')['Age'].agg(Average='mean', Oldest='max', Youngest='min'))

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
       'Salary': [5000, 6000, 7000]}
df2 = pd.DataFrame(data)
# print(df2)
# print(df1)

print(pd.merge(df1,df2,on="Age",how='right'))
# print(pd.concat([df1,df2])) #mix two or more columns
# grp=df1.groupby('City')['Age'].sum()
# print(grp)
# print(df1.join(df2))

# print(pd.pivot_table(df1,values='Age',index='City',aggfunc=sum))
# print(pd.crosstab(df1['Age'],df1['City']))

# print(df1)

# print(pd.crosstab(df1['Name'],df1['Age']))
# print(df1['Age'].apply(lambda x: x**2))   # same like map function

# print(pd.cut(df1["Age"],bins=[0,18,35,60],labels=['Youth','Adult','Senior']))

# date time operations

data = {
    'DateTime': [
        '2025-01-01 08:00:00', '2025-02-01 12:00:00', '2025-01-01 18:00:00',
        '2022-01-02 08:00:00', '2023-03-02 12:00:00', '2021-01-02 18:00:00',
        '2022-01-03 08:00:00', '2025-01-03 12:00:00', '2025-01-03 18:00:00',
        '2025-05-04 08:00:00', '2025-04-04 12:00:00', '2023-01-04 18:00:00'
    ],
    'City': [
        'Delhi', ' Delhi', 'Delhi',
        'Mumbai', 'Mumbai', 'Mumbai',
        'Chennai', 'Chennai', 'Chennai',
        'Delhi', 'Delhi', 'Delhi'
    ],
    'Temperature': [20, 25, 22, 26, 30, 28, 29, 33, 31, 23, 27, 24],
    'Humidity': [40, 45, 42, 50, 55, 52, 60, 65, 63, 48, 53, 50],
    'Sales': [180, 200, 190, 220, 260, 240, 270, 300, 280, 210, 230, 220]
}

df3 = pd.DataFrame(data)
df3['dateTime'] = pd.to_datetime(df3['DateTime'])
# print(df3)



# print(pd.date_range(start='2025-01-01 08:00:00',end='2025-11-03 12:00:00',freq="M"))
# print(pd.date_range(start='2025-01-01 08:00:00',end='2025-11-03 12:00:00',freq="YE-MAR"))

#get year only
# print(df3['dateTime'].dt.year)
# print(df3['dateTime'].dt.month)

#set index
print(df3.set_index("dateTime",inplace=True))
# print(df3.set_index(df3['dateTime'].dt.year))

#resample
# print(df3.resample('YE', on='dateTime').sum())
# print(df3.resample('YE', on='dateTime').max())
# print(df3.resample('M', on='dateTime').sum())
# print(df3.resample('ME', on='dateTime').max())

#shift
# print(df3.shift(periods=3,fill_value=0))  #if convert to date timefill value with zero don't work

#time zone
# print(df3.tz_localize("UTC"))   # for running this index (date time)should there 
# print(df3.tz_localize("GMT")) 
# print(df3.tz_localize("Asia/kolkata")) 
# x=df3.tz_localize("UTC")
# print(x.tz_convert("UTC"))


##string operations
# print(df3['City'].str.upper())
# print(df3['City'].str.lower())
# print(df3['City'].str.strip())
# print(df3['City'].str.replace('Delhi','New Delhi'))

#checking can we use replace for null values

# print(df1.replace(to_replace=[None], value=0))
# print(df1["Name"].replace(to_replace=[None], value=0))

#
# print(df1['Name'].str.contains("bob"))
# print(df1['Name'].str.contains("Alice"))

#
# print(df1["Name"].str.startswith("A"))
# print(df1["Name"].str.endswith("e"))

# 
# print(df1['Name'].str.split('')) #each letter
# print(df1['Name'].str.split())  #each words

#
# print(df1["Name"].str.get(1)) # 1 st index letter

#
# print(df1["Name"].str.join("-"))

#
# regex -> regular expressions
# print(df1["Name"].str.extract('regex'))

#find all
# print(df1['Name'].str.findall(r""))


