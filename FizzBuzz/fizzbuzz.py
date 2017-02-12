import numpy as np
import sys

def answer(value):
    if isinstance(value, int):
        if (value % 3 == 0) and (value % 15 != 0):
            print("Fizz")
        elif (value % 5 == 0) and (value % 15 != 0):
            print("Buzz")
        elif value % 15 == 0:
            print("FizzBuzz")
        else:
            print(value)
    else:
        print("Enter an integer")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        value = sys.argv[1]
        if value:
            value = int(sys.argv[1])
        answer(value)
    else:
        print("Enter an integer")