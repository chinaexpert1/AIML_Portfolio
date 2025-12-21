'''Module 13 philosophers module by Andrew Taylor atayl136'''

import random
import threading
import time

from fork import Fork

# Philsopher Class
class Philosopher(threading.Thread):
    running = True
    
    # Initialize a Philosopher Object
    def __init__(self, name: str, left_fork: Fork, right_fork: Fork):
        super(Philosopher, self).__init__()
        self.name = name
        self.left_fork = left_fork
        self.right_fork = right_fork
    
    # run defines the threads body
    def run(self):
        while self.running == True:
            self.think()
            self.eat()
            print(f'{self.name} is cleaning up.')
    
    # think is like some intial blocking to prevent Deadlock at first
    def think(self):
        thinking = random.uniform(3, 5)
        print(f'{self.name} is thinking for {thinking} seconds')
        time.sleep(thinking)
        print(f'{self.name} is now hungry.')
        
    def eat(self):
        left = self.left_fork.acquire_fork()
        left
        if left == True:
               right = self.right_fork.acquire_fork()
               right 
               if right == True:
                   eating = random.uniform(3, 5)
                   print(f'{self.name} is eating for {eating} seconds.')
                   time.sleep(eating)
                   self.right_fork.release_fork()
                   print(f'{self.name} has put down his right fork.')
               self.left_fork.release_fork()
               print(f'{self.name} has put down his left fork.')
        else: 
            return
           
                
        
        
            
    
        







