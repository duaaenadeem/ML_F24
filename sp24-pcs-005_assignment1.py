#!/usr/bin/env python
# coding: utf-8

# <p style="text-align: right;"> Name : Duaa-e-Nadeem (SP24-PCS-005) </p>

# ## ASSIGNMENT 1

# ### Task 1: Lists, Dictionaries, Tuples

# #### 1.1. Creaye a list: nums = [3, 5, 7, 8, 12], make another list named ‘cubes’ and append the cubes of the given list ‘nums’ in this list and print it.

# In[1]:


nums = [3, 5, 7, 8, 12]
cubes = []

for num in nums:
    # Appending the cube of each number to the cubes list
    cubes.append(num ** 3)  

print("Cubes of the list nums:", cubes)


# #### 1.2. Create an empty dictionary: dict = {}, add the following data to the dictionary: ‘parrot’: 2, ‘goat’: 4, ‘spider’: 8, ‘crab’: 10 as key value pairs.

# In[2]:


dict = {}
dict['parrot'] = 2
dict['goat'] = 4
dict['spider'] = 8
dict['crab'] = 10

print("Dictionary with animals and their number of legs:", dict)


# #### 1.3. Use the ‘items’ method to loop over the dictionary (dict) and print the animals and their corresponding legs. Sum the legs of each animal, and print the total at the end.

# In[3]:


total_legs = 0

for animal in dict:  # Looping directly over the keys (animals)
    legs = dict[animal]  # Accessing the value (legs) using the key
    print(f"The {animal} has {legs} legs.")
    total_legs += legs

print("Total number of legs:", total_legs)


# #### 1.4. Create a tuple: A = (3, 9, 4, [5, 6]), change the value in the list from ‘5’ to ‘8’.

# In[8]:


A = (3, 9, 4, [5, 6])
A[3][0] = 8  # Changing 5 to 8 in the list
print("Changed tuple A:", A)


# #### 1.5. Delete the tuple A.

# In[9]:


del A


# #### 1.6. Create another tuple: B = (‘a’, ‘p’, ‘p’, ‘l’, ‘e’), print the number of occurrences of ‘p’ in the tuple.

# In[10]:


B = ('a', 'p', 'p', 'l', 'e')
print("Occurrences of 'p' in tuple B:", B.count('p'))


# #### 1.7. Print the index of ‘l’ in the tuple.

# In[11]:


print("Index of 'l' in tuple B:", B.index('l'))


# ### Task 2: Numpy

# #### Use built-in functions of numpy library to complete this task.
# #### List of functions available here (https://numpy.org/doc/1.19/genindex.html)
# #### A = 
# ####      1   2      3      4
# ####       5    6     7       8
# ####       9    10  11     12
# #### z = np.array([1, 0, 1])

# #### 2.1 Convert matrix A into numpy array

# In[12]:


import numpy as np

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print("Matrix A as numpy array:")
print(A)


# #### 2.2 Use slicing to pull out the subarray consisting of the first 2 rows and columns 1 and 2. Store it in b which is a numpy array of shape (2, 2).

# In[13]:


b = A[:2, :2]

print("Subarray b (first 2 rows and columns 1 and 2):")
print(b)


# #### 2.3 Create an empty matrix ‘C’ with the same shape as ‘A’.

# In[14]:


C = np.empty_like(A)
print("Empty matrix C with the same shape as A:")
print(C)


# #### 2.4 Add the vector z to each column of the matrix ‘A’ with an explicit loop and store it in ‘C’. Create the following:
# #### X = np.array([[1,2],[3,4]])
# #### Y = np.array([[5,6],[7,8]])
# #### v = np.array([9,10])

# In[15]:


z = np.array([1, 0, 1])

for i in range(A.shape[1]):  
    C[:, i] = A[:, i] + z  # Add vector z to each column of A

print("Matrix C after adding vector z to each column of A:")
print(C)


# #### 2.5 Add and print the matrices X and Y.

# In[17]:


X = np.array([[1, 2],
              [3, 4]])
Y = np.array([[5, 6],
              [7, 8]])

sum_XY = np.add(X, Y)
print("\nSum of X and Y:\n", sum_XY)


# #### 2.6 Multiply and print the matrices X and Y.

# In[18]:


mul_XY = X * Y 
print("\n multiplication of X and Y:")
print(mul_XY)


# #### 2.7 Compute and print the element wise square root of matrix Y.

# In[19]:


sqrt_Y = np.sqrt(Y) 
print("\n Element-wise square root of matrix Y:")
print(sqrt_Y)


# #### 2.8 Compute and print the dot product of the matrix X and vector v. 

# In[20]:


v = np.array([9, 10])

dot_product = np.dot(X, v) 
print("\n Dot product of matrix X and vector v:")
print(dot_product)


# #### 2.9 Compute and print the sum of each column of X.

# In[21]:


column_sum_X = np.sum(X, axis=0) 
print("\n Sum of each column of matrix X:")
print(column_sum_X)


# ### Task 3: Functions and Loops
# #### 3.1 Create a function ‘Compute’ that takes two arguments, distance and time, and use it to calculate velocity. 

# In[23]:


def Compute(distance, time):
    if time == 0:
        return "Time cannot be zero!"  
    velocity = distance / time  # Velocity = Distance / Time
    return velocity


# #### 3.2 Make a list named ‘even_num’ that contains all even numbers up till 12. Create a function ‘mult’ that takes the list ‘even_num’ as an argument and calculates the products of all entries using a for loop.

# In[25]:


even_num = [2, 4, 6, 8, 10, 12]  
def mult(numbers):
    product = 1  
    for num in numbers:
        product *= num  
    return product


# ### Task 4: Pandas
# ### Create a Pandas dataframe named ‘pd’ that contains 5 rows and 4 columns, similar to the one given below:
# ### C1 C2 C3 C4
# ### 1 6 7 7
# ### 2 7 9 5
# ### 3 5 8 2
# ### 5 4 6 8
# ### 5 8 5 8
# #### 4.1 Print only the first two rows of the dataframe.

# In[26]:


import pandas as pd
# Create the dataframe
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}
df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df)

print("\nFirst two rows:")
print(df.head(2))


# #### 4.2 Print the second column.

# In[28]:


print("Second column (C2):")
print(df['C2']) 


# #### 4.3 Change the name of the third column from ‘C3’ to ‘B3’.

# In[30]:


df.rename(columns={'C3': 'B3'}, inplace=True)
print("DataFrame after renaming column C3 to B3:")
print(df)


# #### 4.4 Add a new column to the dataframe and name it ‘Sum’.

# In[31]:


df['Sum'] = 0 
print("DataFrame after adding 'Sum' column:")
print(df)


# #### 4.5 Sum the entries of each row and add the result in the column ‘Sum’.

# In[32]:


df['Sum'] = df.sum(axis=1) 
print("DataFrame after summing rows into 'Sum' column:")
print(df)


# #### 4.6 Read CSV file named ‘hello_sample.csv’ (the file is available in the class Google Drive shared folder) into a Pandas dataframe.

# In[33]:


csv_df = pd.read_csv('hello_sample.csv')


# #### 4.7 Print complete dataframe.

# In[34]:


print("Complete dataframe from CSV:")
print(csv_df)


# #### 4.8 Print only bottom 2 records of the dataframe.

# In[35]:


print("Bottom two records of the dataframe:")
print(csv_df.tail(2))


# #### 4.9 Print information about the dataframe.

# In[36]:


print("Information about the dataframe:")
print(csv_df.info()) 


# #### 4.10 Print shape (rows x columns) of the dataframe.

# In[37]:


print("Shape of the dataframe (rows x columns):")
print(csv_df.shape)


# #### 4.11 Sort the data of the dataFrame using column ‘Weight’.

# In[39]:


if 'Weight' in csv_df.columns: 
    sorted_df = csv_df.sort_values(by='Weight')
    print("Data sorted by 'Weight':")
    print(sorted_df)
else:
    print("\n'Weight' column not found in the dataframe.")


# #### 4.12 Use isnull() and dropna() methods of the Pandas dataframe and see if they produce any changes.

# In[40]:


print("Checking for missing values using isnull():")
print(csv_df.isnull()) 

print("\nDropping rows with missing values using dropna():")
clean_df = csv_df.dropna() 
print(clean_df)


# In[ ]:




