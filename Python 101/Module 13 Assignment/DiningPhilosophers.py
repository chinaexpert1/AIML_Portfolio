'''Dining Philsophers client by Andrew Taylor atayl136'''

import time
from fork import Fork
from philosopher import Philosopher


# Dining Philosophers function
def DiningPhilosophers():
    names = ['Plato', 'Aristotle', 'Buddha', 'Marx', 'Nietzsche']

    # create forks with list comprehension
    fork1, fork2, fork3, fork4, fork5 = [Fork() for name in names]
    forklist = [fork1, fork2, fork3, fork4, fork5, fork1]    # list of forks modified for loop simplicity
    
    # build a tuple list to generate the objects
    philtuplelist = []
    for i in range(len(names)):
        philtuplelist.append((names[i], forklist[i], forklist[i+1]))
    
    # create philosopher objects from list comprehension
    philosopherobjects = [Philosopher(*x) for x in philtuplelist]
    
    # name the objects just for readability
    Plato, Aristotle, Buddha, Marx, Nietzsche = [x for x in philosopherobjects]
    philosophers = Plato, Aristotle, Buddha, Marx, Nietzsche
    
    # start threads
    for i in range(len(philosophers)):
        philosophers[i].start()

    # give the program time to run
    time.sleep(10)
    # stop the objects
    for i in range(len(philosophers)):
        philosophers[i].running = False

    # terminate the threads
    for i in range(len(philosophers)):
        philosophers[i].join()
        


if __name__ == '__main__':
    DiningPhilosophers()




