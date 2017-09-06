from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from root_pandas import read_root
import sys
from itertools import islice
import readline as rd

# https://stackoverflow.com/questions/2533120/show-default-value-for-editing-on-python-input-possible
def rlinput(prompt, prefill=''):
   rd.set_startup_hook(lambda: rd.insert_text(prefill))
   try:
      return input(prompt)
   finally:
      rd.set_startup_hook()

a1 = input("Scrivi il primo addendo: ")
a2 = input("Scrivi il secondo addendo: ")
# a1 = float(input("Scrivi il primo addendo: "))
# a2 = float(input("Scrivi il primo addendo: "))

print("a1+a2: ", a1+a2)

b1 = float(rlinput("Scrivi il primo addendo (rd): ", str(a1)))
# rd.redisplay()
# b1 = float(input())
# print(b1)

b2 = float(rlinput("Scrivi il primo addendo (rd): ", str(a2)))


print("b1+b2: ", b1+b2)
