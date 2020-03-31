import math

# function converting to upper class
def convert(str):
    temp=""
    for i in str:
        if ord(i) > ord('a') and ord(i) < ord('z'):
            i = chr(ord(i)-32)
        temp+=i
    return temp

# function to return the sin and cos of an angle
def getTriangle(alpha):
    alpha = math.radians(alpha)
    return math.sin(alpha), math.cos(alpha)

# function to test dictionary
def testDict():
    dict = {}
    dict[1] = 1
    dict[2] = 2
    print(dict)

# function to test get all values of all columns
def testCol(data):
    col_data = []
    col_num = len(data[0])
    for col in range(col_num):
        current_col = [row[col] for row in data]
        col_data.append(current_col)
    return col

# function to test the function list.count()
def testConut(data, value):
    return data.count(value)

