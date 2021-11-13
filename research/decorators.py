
def doubler(x):
    """This function will double the number"""
    return x*2

# print(doubler)
# print(doubler(9))
# print(doubler.__doc__)
# print(doubler.__name__)
# print(dir(doubler))

# decorator starts here

def info(funct):
    def wrapper(*arg):
        print(f"Function Name : {funct.__name__}")
        print(f"Doc Name : {funct.__doc__}")

        return funct(*arg)
    return wrapper

@info
def doubler(x):
    """This function will double the number"""
    return x*2

# my_decorator = info(doubler)
# print(my_decorator(9))

# print(doubler(6))

# Stack two decorators 

def bold(func):
    def wrapper():
        return "<b>" + func() + "<b>"
    return wrapper

def italic(func):
    def wrapper():
        return "<i>" + func() + "<i>"
    return wrapper

@bold
@italic
def text_format():
    return "Python Rocks!!!"

print(text_format())



