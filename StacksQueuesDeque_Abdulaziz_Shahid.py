#Problem 1

class Node:
    def __init__(self,data):
        self.data = data
        self.next = None


class Stack():
    def __init__(self):    
        self.head = None
        
    def isEmpty(self):       
        return self.head == None
             
    
    def push(self,data):
        
        if self.head == None: #if to check if the head is empty and if it is then the new entered data is the head
            self.head = Node(data)
             
        else:  #if the head is not empty then it creates a new node and saves the previous node aka the head as the next reference. Then makes the the new node the head 
            newNode = Node(data)
            newNode.next = self.head
            self.head = newNode
        print(f' Adding {self.head.data} to the stack.\n')
            

            
    def pop(self):
        if self.isEmpty() == None: #can't pop an empty list
            return None
        else:
            removedNode = self.head #removing the head of the stack aka the node
            self.head = self.head.next #setting the new head of the stack based off of the next node in the link
            removedNode.next = None #there is no next node for the removed one
            print( f'Removing {removedNode.data} from the stack!' )
        
        
    def peek(self): #to find the head of out data
        if self.isEmpty():
            return None
        else:
            return self.head.data
        
    def show(self):
        
        topNode = self.head
        if self.isEmpty():
            print("Stack is empty")
        else:
            print('Printing Entire Stack.')
            while(topNode != None): #need to have None condition so the system knows it is at the end of the link list
                print(topNode.data)
                topNode = topNode.next #so it can keep calling the next node in the sequence
            return
        

#Problem 2

class Queue:
    
    def __init__(self):
        
        self.start = []
        self.end = []
        self.count = 0
        
    def isEmpty(self):
        return self.start == []
    
    def enQueue(self,place):
        queHolder = Node(place)
        print(f'Adding {place} to que.')
        if self.end == []:
            self.start = queHolder
            self.end = queHolder
            
        else:
            self.end.next = queHolder
            self.end = queHolder
        
        self.count += 1
        print(f'Total queue size is now {self.count}. \n')
        

                             
    def deQueue(self):
        if self.isEmpty() == []:
            print('There is nothing left to remove')
    
        queHolder = self.start
        self.start = queHolder.next
        
        if(self.start == []):
            self.end = []
            
        self.count -= 1
        
        return (print(f'Removing {queHolder.data} from queue.'),
                print(f'Next in line is {self.start.data}.'),
                print(f'Total waiting is now {self.count}.\n'))
    
    def size(self):
        return print(f'The Queue size is {self.count}')
        
#Problem 3
    
class deQue:
     
    def __init__(self):
         self.front = None
         self.rear = None
         self.count = 0
         
    def isEmpty(self):
        return self.front == []
    
    def frontInsertion(self,place):
        queHolder = Node(place)
        print(f'Adding {place} to front of the queue.')
        
        if self.front == None:
            self.front = queHolder
            
        else:
            queHolder.next = self.front
            self.front = queHolder
        
        self.count += 1
        print(f'Total queue size is now {self.count}. \n')
        
    def rearInsertion(self,place):
        queHolder = Node(place)
                       
        if self.front == None:
            self.front = queHolder
            self.rear = queHolder
        
        else:
            hold = self.front
            while hold.next != None:
                hold = hold.next
            hold.next = queHolder
                      
        self.count += 1
        
        return (print(f'Inserting {queHolder.data} to the back of the queue.'),
                print(f'Total waiting is now {self.count}.\n'))
    
          
    def deleteFront(self):
               
        if self.isEmpty() == []:
            print('There is nothing left to remove')
        else:
            queHolder = self.front
            self.front = self.front.next
            del queHolder
            print('Front que has been deleted \n')
        self.count -= 1
                
    def deleteRear(self):    
        
        if self.isEmpty() == []:
            print('There is nothing left to remove')
            
        else:
            hold=self.front
            prev = None
            while hold.next!=None:
                prev=hold
                hold=hold.next
            prev.next=hold.next
            del hold
            print('Rear que has been deleted \n')
        self.count -= 1
            
    def show(self):
        topNode = self.front
        
        if self.isEmpty():
            print("Stack is empty")
        else:
            print('Printing Entire List.')
            while(topNode != None): #need to have None condition so the system knows it is at the end of the link list
                print(topNode.data)
                topNode = topNode.next #so it can keep calling the next node in the sequence
            return
        
        
        
                

         
testDeque = deQue()
testDeque.frontInsertion(1)
testDeque.frontInsertion(2)
testDeque.frontInsertion(3)         
testDeque.rearInsertion(9)     
testDeque.rearInsertion(6)
testDeque.show()
testDeque.deleteFront()
testDeque.show()       
testDeque.deleteRear()
testDeque.show() 


#Testing my stackeed linked list below
        
testStack = Stack()
testStack.push(1)
testStack.push(2)
testStack.push(3)        
testStack.push(4)
print(testStack.pop())
print(testStack.pop())

testStack.show()


#Problem 2

testQueue = Queue()
testQueue.enQueue(1)
testQueue.enQueue(2)
testQueue.enQueue(3)
testQueue.enQueue(4)
testQueue.deQueue()
testQueue.deQueue()
testQueue.size()
#testQueue.show()