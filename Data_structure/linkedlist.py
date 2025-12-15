# Node class for Linked List
class Node:
    def __init__(self, data):
        self.data = data   # stores the value
        self.next = None   # points to the next node


# Linked List class
class LinkedList:
    def __init__(self):
        self.head = None   # start of the linked list

    # Insert at the end
    def insert(self, data):
        new_node = Node(data)
        if self.head is None:   # if list is empty
            self.head = new_node
            return
        temp = self.head
        while temp.next:   # traverse to the last node
            temp = temp.next
        temp.next = new_node

    # Display linked list
    def display(self):
        if self.head is None:
            print("List is empty")
            return
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

    # Delete a node by value
    def delete(self, key):
        temp = self.head

        # Case 1: head node itself contains the key
        if temp is not None and temp.data == key:
            self.head = temp.next
            temp = None
            return

        # Case 2: search for the key
        prev = None
        while temp is not None and temp.data != key:    
            prev = temp
            temp = temp.next

       
        # Key not found
        if temp is None:
            print("Value not found!")
            return

         # Unlink the node
        prev.next = temp.next
        temp = None
        


# Example usage
# if __name__ == "__main__":
llist = LinkedList()

llist.insert(input("enter a value: "))
llist.insert(20)
llist.insert(30)
print("Linked List:")
llist.display()

print("Deleting 20...")
llist.delete(20)
llist.display()
