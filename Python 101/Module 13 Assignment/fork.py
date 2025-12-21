'''Module 13 Fork module by Andrew Taylor atayl136'''

import threading

class Fork:
    def __init__(self):
        self.lock = threading.Lock()
    
    def acquire_fork(self):
        if self.lock.locked() == False:
            self.lock.acquire()
            return True
        return False
    
    def release_fork(self):
        self.lock.release()



