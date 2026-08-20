import time
from utils2.timeout import timeout

@timeout(5)
def delay_func(delay=5): 
    time.sleep(delay) 

def main():
    for to in range(2,8):
        print(f'Running with timed out {to}s: ',end="")
        try:
            delay_func(to)
            print(f'Success!')
        except:
            print(f'Timed out')

if __name__ =="__main__":
    main()