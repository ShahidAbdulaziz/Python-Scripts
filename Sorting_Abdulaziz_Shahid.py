#Consider the following list of integers: [1,2,3,4,5,6,7,8,9,10].  Show how this list is sorted by the following algorithms:
#bubble sort
#selection sort
#insertion sort


# A buble sorta sorts a list by fining if the current number is smaller than the next number in the list. If it is, then it swaps the numbers
listTest = [64, 34, 25, 12, 22, 11, 90]

def bubbleSort(Intlist):
    length = len(Intlist)
 
    for i in range(length-1): #Lenght of list minus 1 to account for a ind starting at 0

        for number in range(0, length-i-1):

            if Intlist[number] > Intlist[number + 1] : # Plus one 
                Intlist[number], Intlist[number + 1] = Intlist[number + 1], Intlist[number]
 

bubbleSort(listTest)
print(listTest)

#Selection Sort

def selectionSort(Intlist):
    length = len(Intlist)
    
    for number in range(length):
          
        # Find the minimum element in remaining 
        # unsorted array
        minNum = number
        for currentNum in range(number+1, length):
            if Intlist[minNum] > Intlist[currentNum]:
                minNum = currentNum
                  
        # Swap the found minimum element with 
        # the first element        
        Intlist[number], Intlist[minNum] = Intlist[minNum], Intlist[number]
        
listTest = [64, 34, 25, 12, 22, 11, 90]
selectionSort(listTest)
print(listTest)
        
        
        
def insertionSort(Intlist):
    length = len(Intlist)
    # Traverse through 1 to len(arr)
    for number in range(1, length):
  
        key = Intlist[number]
  
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        currentNum = number-1
        while currentNum >=0 and key < Intlist[currentNum] :
                Intlist[currentNum+1] = Intlist[currentNum]
                currentNum -= 1
        Intlist[currentNum+1] = key
  
  
# Driver code to test above
listTest = [64, 34, 25, 12, 22, 11, 90]
insertionSort(listTest)
print(listTest)


# A list is an order sequence while a dictionary are unordered sets that can be accessed by a key
# A dictionary is coded with the key and the value associated with it while a list just contains a sequence of objects
# A dictionary is used when you want to have an index of values associated with a key for quick lookup. Like if you wanted to know the colors associated with the bikes that are for sale

bikeList = ['Tornado','Cyclone','Rocket'] #list
bikeDictionary = {'Tornado': 'Blue','Cyclone': 'Red','Rocket': 'Orange'}

print(bikeList)
print(bikeDictionary['Tornado'])

