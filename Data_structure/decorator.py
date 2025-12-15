def add_cheese(func):
    def wrapper():    # wrapper is a std function name 
        print("Added Cheese")
        func()             # calling shawarma function , below code will run after this 
        print("Cheesy item  is ready")
        print("----------------------------")
    return wrapper


@add_cheese
def shawarma():
    print("Shawarma is ready")

shawarma()

# if we want to apply wrapper on different function call it seperately
@add_cheese
def maggie():
    print("maggie is ready")

maggie()