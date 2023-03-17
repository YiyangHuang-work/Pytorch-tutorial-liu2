class Foobar:
    def __init__(self):
        pass

    def __call__(self,*args,**kwargs):
        print("Hello" + str(args[0]))

def func(*args,**kwargs):
    print(args)
    print(kwargs)

func(1,2,3,4,x=3,y=5)
foobar = Foobar()
foobar(1,2,3)

