import os, errno, math
import functools
from typing import Callable, TypeVar

def silentremove(filename):
    try:
      os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
      if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
        raise # re-raise exception if a different error occurred
          
          
def clear_dir(directory):
  """
  Recursively clears the directory
  """
  for f in os.listdir(directory):
    path = os.path.join(directory, f)
    if os.path.isdir(path):
      clear_dir(path)
      os.rmdir(path)  # Remove the now-empty directory
    else:
      silentremove(path)
          
def mkdir_empty(directory):
  """
  Ensures the directory exists and is empty
  """
  os.makedirs(directory, exist_ok=True)
  clear_dir(directory)



def is_prime(number):
    if number < 2:
        return False
    if number in (2, 3):
        return True
    if number % 2 == 0 or number % 3 == 0:
        return False

    limit = math.isqrt(number)
    for i in range(5, limit + 1, 6):
        if number % i == 0 or number % (i + 2) == 0:
            return False
    return True


T = TypeVar('T')

def iife(func: Callable[[], T]) -> T:
    """
    Decorator to turn a function into an immediately invoked function expression (IIFE).
    Example usage:
    @iife
    def my_function():
        # Your code here
        return result
        
    result = my_function
    
    """
    return func()