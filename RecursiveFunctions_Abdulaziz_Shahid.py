# 1: Write a short recursive Python function that finds the minimum and maximum values in a sequence without using any loops.
# 2: Write a recursive function to reverse a list.


import sys

#Functions for program below

def maxLocator(listValues):
    
    listValues.sort()
    
    if len(listValues) == 1:
        return listValues[0]

    else:
        valueCheck = maxLocator(listValues[1:])


        return valueCheck if valueCheck > listValues[0] else listValues[0]
    
    
def minLocator(listValues):
    
    listValues.sort()
    
    if len(listValues) == 1:
        return listValues[0]

    else:
        valueCheck = minLocator(listValues[1:])


        return listValues[0] if listValues[0] < valueCheck  else valueCheck

def reverseList(listValues):
    
    if len(listValues) == 0:
         return []
        
    return [listValues[-1:]] + reverseList(listValues[:-1])
    
#Program Below

while True:

    elementCnt = input("Enter the whole numbers in your sequence seperated by a space: ")
    numList = elementCnt.split()
    numList = list(map(int,numList))
    print("The numbers you picked are: ", numList)

    

    while True:
        findMax = input("Would you like to find the maximum number for that list? (Yes/No) ").lower()
        
        if findMax == "yes":
            print(f"The maximum number in your list is {maxLocator(numList)}" )
            break
            
        elif findMax == "no":
            break
        else:
            print("Answer Must Be 'Yes' or 'No'")

        

    while True: 
        findMin = input("Would you like to find the minimum number for that list? (Yes/No)").lower()
    
    
        if findMin == "yes":
            print(f"The mimimum number in your list is {minLocator(numList)}" )
            break
        elif findMin == "no":
            break
        else:
            print("Answer Must Be 'Yes' or 'No'")
            
    while True:
        reverseListInput = input("Would you like to see your list in reverse order? (Yes/No) ").lower()
        
        if reverseListInput == 'yes':
            print(f'The reverse of your list {numList} is {reverseList(numList)} ')
            break
        elif reverseListInput == 'no':
            break
        else:
            print("Please only enter yes or no!")
        
    while True:     
        repeatProgram = input("Would you like to input a new list of numbers? (Yes/No) ").lower()
        
        if repeatProgram == "yes":
            break
           
        elif repeatProgram == 'no':
            print("Exiting Program")
            sys.exit()
        else:
            print("Answer Must Be 'Yes' or 'No'")
        
    
    
    
