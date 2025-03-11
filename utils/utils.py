import os, errno

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
          
def mkdir_empty(directory):
  """
  Ensures the directory exists and is empty
  """
  os.makedirs(directory, exist_ok=True)
  for f in os.listdir(directory):
    silentremove(os.path.join(directory, f))
