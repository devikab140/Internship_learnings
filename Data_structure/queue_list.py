Queue=[]

def enqueue(item):
    Queue.append(item)

def dequeue():
    if not Queue:
        print("Queue is empty!")
        return None
    return Queue.pop(0)      #inex is 0 since we are implementing in list first one will be 0 th inex and can remove easily

enqueue(10)
enqueue(20)
dequeue()
print(Queue)