'''
Regular Polygon Animation

author: Lars A. Gehrke
'''

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse

class Dot():
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def _list(self):
        return [self.x, self.y]

    def __repr__(self):
        return str(self._list())

    @staticmethod
    def direction(a,b):
        return Dot(b.x-a.x, b.y-a.y)

    @staticmethod
    def normalize(dot):
        length = dot.length()
        return Dot(dot.x / length, dot.y / length)

    def clone(self):
        return Dot(self.x,self.y)

    def direction(self,target):
        return Dot(target.x-self.x, target.y-self.y)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)


    def move_to_dot(self,target, speed):
        diff = self.direction(target)
        diff_norm = diff.normalize(diff)
        self.x = self.x + diff_norm.x * speed
        self.y = self.y + diff_norm.y * speed

    def move_by_angle(self, theta, speed):
        angle = math.radians(theta)
        u = Dot(math.cos(angle), math.sin(angle))
        self.x = self.x + u.x * speed
        self.y = self.y + u.y * speed


class MovingRegularPolygon():
    def __init__(self, n, distance, speed, max_sim_steps = None):
        if(n < 3):
            sys.exit("ERROR: The amount of corners of the Polygon should be 3 or higher.")
        self.n = n
        self.inner_angles = 180 * (n-2)
        self.inner_angle = self.inner_angles / n
        self.outter_angle = 180 - self.inner_angle

        self.distance = distance
        self.speed = speed
        self.simulation_step = 0
        self.max_sim_steps = max_sim_steps
        # this is the archive of all polygons in the past
        # a list of Dot lists
        self.simulation = []

        # this is the current polygon
        # a list of Dot objects
        self.polygon = [Dot(0,0) for i in range(n)]
        self.create_starting_points()
        
    def create_starting_points(self):
        self.polygon[0] = Dot(0,0)
        self.polygon[1] = Dot(self.distance, 0)
        theta = 0
        for i in range(2,self.n):
            theta = 360 - self.outter_angle * (i-1)
            clone = self.polygon[i-1].clone()
            clone.move_by_angle(theta, self.distance)
            self.polygon[i] = clone

    def x_values(self):
        return [p.x for p in self.polygon]

    def y_values(self):
        return [p.y for p in self.polygon]

    def x_hist_values(self):
        return [p.x for t in range(len(self.simulation)) for p in self.simulation[t]]

    def y_hist_values(self):
        return [p.x for t in range(len(self.simulation)) for p in self.simulation[t]]


    def simulate(self):
        if self.max_sim_steps is None or \
        self.simulation_step < self.max_sim_steps:
            self.simulation.append([Dot(p.x,p.y) for p in self.polygon])
            for i in range(self.n):
                self.polygon[i].move_to_dot(self.polygon[(i+1) % self.n], self.speed)
            self.simulation_step += 1

        return self.simulation_step
        

    def plot_polygon(self):
        x = self.x_values()
        y = self.y_values()

        theta = 360 - self.outter_angle * (2-1)
        plt.scatter(x, y)
        plt.show()
       

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--corners', type=int, default = 3,
    help='Number of corners of the regular polygon.')
parser.add_argument('-d', '--distance', type=int, default = 12,
    help='Length of the edges of the regular polygon.')
parser.add_argument('-s', '--speed', type=float, default = 0.1,
    help='Speed of the point movement to each other.')
parser.add_argument('-t', '--time-steps', type=int,
    help='Amount of simulation steps that should be rendered.')
parser.add_argument('-i', '--interval', type=int, default = 20,
    help='Interval parameter of the animation.')
options = parser.parse_args()
        
polygon = MovingRegularPolygon(options.corners,options.distance, options.speed, 
    max_sim_steps= options.time_steps)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
d, = plt.plot(polygon.x_values(),
             polygon.y_values(), 'ro')

hist_x, hist_y = [],[]

d2, = plt.plot(hist_x,
             hist_y, 'ko', markersize=3)
# circle = plt.Circle((5, 5), 1, color='b', fill=False)
# ax.add_artist(circle)


# animation function.  This is called sequentially
def animate(i):
    hist_x.append(polygon.x_values())
    hist_y.append(polygon.y_values())
    sim_step = polygon.simulate()
    d.set_data(polygon.x_values(),
               polygon.y_values())
    d2.set_data(hist_x,
             hist_y)
    plt.title(f"Simulation step {str(sim_step)}")
    return d,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=options.time_steps, interval=options.interval)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save(f"animate_regular_polygon_{options.corners}.mp4", fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
