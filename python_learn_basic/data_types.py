
# list
fruits=["Apple","Orange","Grape"]
print(fruits)

#index 
print(fruits.index("Orange"))

#append
fruits.append("Kiwi")
print(fruits)
#insert
fruits.insert(1,"Lemon")
print(fruits)
#remove
fruits.remove("Orange")
print(fruits)
#pop
fruits.pop(1)
print(fruits)
# type and count()
print(type(fruits))
print(fruits.count("Apple"))

#clear
fruits.clear()
print(fruits)

### tuple

colours=("Black","Blue","Green","Blue")
print(colours)
print(type(colours))
# count() and index()
print(colours.count("Blue"))
print(colours.index("Blue"))


#changing to list
clr=list(colours)
print(clr)
clr.append("white")
colours=tuple(clr)
print(colours)

#set
foods={"Biriyani","Icecream","Choclate","Biriyani"}
print(foods)

#add()   , we are using seperate print
foods.add("Kulfi")
print(foods)

#remove
foods.remove("Icecream")
print(foods)

#discard
foods.discard("Kulfi")
print(foods)

foods.discard("Egg")
print(foods)

#pop() --- remove the first elmnt since the set has no specific indx nmbr
foods.pop()
print(foods)
#clear
foods.clear()
print(foods)

#set operations
s1={1,2,3,4}
s2={4,7,9,10}

print(s1.union(s2))
print(s1|s2)

print(s1.intersection(s2))
print(s1&s2)

print(s1.difference(s2))
print(s1-s2)

print(s1.symmetric_difference(s2))
print(s1^s2)

#dictionary
persons={"name":"devika","age":23,"place":"Kasaragod",3:"abc"}
print(persons)

print(persons.get("name")) #here one o/p is there to show

#dictionary can only update with any other dictionary
person1={"hobbie":"singing",6:"hsl"}
persons.update(person1) #no o/p only updating
print(persons)

print(persons.pop(3))
# print(persons.pop("abc")) , pop not work with value

print(persons.keys())
print(persons.values())
print(persons.items())