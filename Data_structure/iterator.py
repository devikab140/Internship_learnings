# nums = [1, 2, 3]
# #print(nums[0])


# #nums_it=iter(nums)
# # print(dir(nums_it))


# # my_list = [1, 2, 3] 
# # iterator = iter(my_list)                      # Calls __iter__() 
# # while True: 
# #      try: 
# #           item = next(iterator)                # Calls __next__()
# #           print(item) 
# #      except StopIteration: 
# #           break



# class CountDown:
#       def __init__(self, start):
#              self.start = start
#       def __iter__(self): 
#               return self 
#       def __next__(self):
#               if self.start <= 0: 
#                     raise StopIteration 
#               current = self.start 
#               self.start -= 1 
#               return current 

# Usage 

# val=CountDown(4)
# # print(next(val))
# # print(next(val))

# for num in CountDown(3):
#         print(num) 


nums=[1,2,3]
print(nums[0])
for i in nums:
    print(i)


it_nums=iter(nums)
print(next(it_nums))
for num in it_nums:
    print(num)