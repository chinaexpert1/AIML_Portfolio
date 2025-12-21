import math

# Solution for the N Body Problem
# I will be following the general structure of the psuedo code
# with my own interpretation and solutions interspersed

# N is for Number of bodies
N = 5

# G is for gravity
G = 6.67E-11

# Index for each body
EARTH = 0
MARS = 1
MERCURY = 2
SUN = 3
VENUS = 4

# variables for the length of the simulation loop
t = 157788000 # total time of simulation
dt = 25000 # time delta
t_total = 0

# input lists
earth = [1.4960e+11, 0.0000e+00, 0.0000e+00, 2.9800e+04, 5.9740e+24]
mars = [2.2790e+11, 0.0000e+00, 0.0000e+00, 2.4100e+04, 6.4190e+23]
mercury = [5.7900e+10, 0.0000e+00, 0.0000e+00, 4.7900e+04, 3.3020e+23]
sun = [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9890e+30]
venus = [1.0820e+11, 0.0000e+00, 0.0000e+00, 3.5000e+04, 4.8690e+24]

# list of lists
planets = [earth, mars, mercury, sun, venus]

# lists for tracking calculation elements
# px is for x position
px = [planets[EARTH][0], planets[MARS][0], planets[MERCURY][0], planets[SUN][0], planets[VENUS][0]]
# py is for y position
py = [planets[EARTH][1], planets[MARS][1], planets[MERCURY][1], planets[SUN][1], planets[VENUS][1]]
# vx is for x velocity
vx = [planets[EARTH][2], planets[MARS][2], planets[MERCURY][2], planets[SUN][2], planets[VENUS][2]]
# vy is for y velocity
vy = [planets[EARTH][3], planets[MARS][3], planets[MERCURY][3], planets[SUN][3], planets[VENUS][3]]
# m is for mass
m = [planets[EARTH][4], planets[MARS][4], planets[MERCURY][4], planets[SUN][4], planets[VENUS][4]]

# now that all the pieces are established, the calculations
# while loop runs the calculation while t_total<t
while t_total < t:
    for i in range(N):
        if i == SUN:
            pass
        else:
            # calculate components
            radius = (((px[SUN]-px[i])**2)+ (py[SUN]-py[i])**2)**0.5
            pairForce = G*((m[SUN]*m[i])/(radius**2))
            Fx = pairForce*((px[SUN]-px[i])/radius)
            Fy = pairForce*((py[SUN]-py[i])/radius)
            ax = Fx/m[i]
            ay = Fy/m[i]
            # new velocities
            vx[i] = vx[i]+ax*dt
            vy[i] = vy[i]+ay*dt
            # new positions
            px[i] = px[i]+vx[i]*dt
            py[i] = py[i]+vy[i]*dt
    # increment loop
    t_total += dt
        
         
# updating nested list for output format
dtEarth = [px[0], py[0], vx[0], vy[0], m[0]]
dtMars = [px[1], py[1], vx[1], vy[1], m[1]]
dtMercury = [px[2], py[2], vx[2], vy[2], m[2]]
dtSun = [px[3], py[3], vx[3], vy[3], m[3]]
dtVenus = [px[4], py[4], vx[4], vy[4], m[4]]      

# print output line by line to four decimal places in scientific notation with NO COMMAS
print(f"{dtEarth[0]:.4e}" + " " + f"{dtEarth[1]:.4e}" + " " + f"{dtEarth[2]:.4e}" + " " + f"{dtEarth[3]:.4e}" + " " + f"{dtEarth[4]:.4e}\n" 
          f"{dtMars[0]:.4e}" + " " + f"{dtMars[1]:.4e}" + " " +f"{dtMars[2]:.4e}" + " " +f"{dtMars[3]:.4e}" + " " +f"{dtMars[4]:.4e}\n" 
              f"{dtMercury[0]:.4e}" + " " + f"{dtMercury[1]:.4e}" + " " + f"{dtMercury[2]:.4e}" + " " + f"{dtMercury[3]:.4e}" + " " + f"{dtMercury[4]:.4e}\n"   
                  f"{dtSun[0]:.4e}" + " " + f"{dtSun[1]:.4e}" + " " + f"{dtSun[2]:.4e}" + " " + f"{dtSun[3]:.4e}" + " " + f"{dtSun[4]:.4e}\n"
                      f"{dtVenus[0]:.4e}" + " " + f"{dtVenus[1]:.4e}" + " " + f"{dtVenus[2]:.4e}" + " " + f"{dtVenus[3]:.4e}" + " " + f"{dtVenus[4]:.4e}")