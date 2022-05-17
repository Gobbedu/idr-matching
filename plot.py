import matplotlib.pyplot as plt
from random import random, randrange
import numpy as np

class bov_plot:
    def __init__(self, size=10, ticks=1, title=""):
        self.fig, self.ax = plt.subplots(1)
        self.size = size
        
        # Select length of axes and the space between tick labels
        self.xmin, self.xmax, self.ymin, self.ymax = -self.size, self.size, -self.size, self.size
        self.ticks_frequency = ticks    

        # Set identical scales for both axes
        self.ax.set(xlim=(self.xmin-1, self.xmax+1), ylim=(self.ymin-1, self.ymax+1), aspect='equal')

        # Set bottom and left spines as x and y axes of coordinate system
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')

        # Remove top and right spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Create 'x' and 'y' labels placed at the end of the axes
        self.ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
        self.ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

        # Create custom major ticks to determine position of tick labels
        x_ticks = np.arange(self.xmin, self.xmax+1, self.ticks_frequency)
        y_ticks = np.arange(self.ymin, self.ymax+1, self.ticks_frequency)
        self.ax.set_xticks(x_ticks[x_ticks != 0])
        self.ax.set_yticks(y_ticks[y_ticks != 0])

        # Create minor ticks placed at each integer to enable drawing of minor grid
        # lines: note that this has no effect in this example with ticks_frequency=1
        self.ax.set_xticks(np.arange(self.xmin, self.xmax+1), minor=True)
        self.ax.set_yticks(np.arange(self.ymin, self.ymax+1), minor=True)

        # Draw major and minor grid lines
        self.ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
            
        # Increase size of plot
        self.tam = 7
        self.fig.set_size_inches(self.tam ,self.tam)
        self.fig.set_dpi(100)
        # self.fig.subplots_adjust(right=0.8)

        plt.suptitle(title)
   
    
    def plot_ransac(self, inlier, outlier, pt2):
        # plot inliers
        in_x = []
        in_y = []
        for pt in inlier:
            in_x.append(pt[0])
            in_y.append(pt[1])
        self.ax.scatter(in_x, in_y, color='g', label="inliers") 

        # plot outliers        
        out_x = []
        out_y = []
        for pt in outlier:
            out_x.append(pt[0])
            out_y.append(pt[1])
        self.ax.scatter(out_x, out_y, color="b", label="outliers")

        # indicate selected subset
        self.ax.scatter([pt2[0][0], pt2[1][0]], [pt2[0][1], pt2[1][1]], color="r", label="subset")
        self.ax.legend(loc="upper left")
        
        # calculate line y = ax + b
        # polyfit [x][y]
        p1 = pt2[0]
        p2 = pt2[1]
        coef = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
        poly = np.poly1d(coef)
        x_ax = np.linspace(-self.size, self.size, 100)
        y_ax = poly(x_ax)
        line = (x_ax, y_ax)
        
        # draw line over subset
        self.ax.plot(line[0], line[1], "r--")   
        
        
    def plot_data(self, data):
        """Take as input data in the form of an iterable type containing cartesian poits as it values
            (x, y) or [x, y]
        """
        x = []
        y = []
        for pt in data:
            x.append(pt[0])            
            y.append(pt[1])
            
        self.ax.scatter(x, y)


    def rand_line(self, x1, y1, x2, y2, n):
        slope = (y2 - y1)/(x2 - x1)
        pts = []

        for i in range(n):
            x = (x2 - x1)*random() + x1 
            y = slope*(x - x1) + y1  
            pts.append((x + random()*2, y + random()*2))
            
        for i in range(n - round(n/4)):
            aux = (randrange(x1, x2), randrange(y1, y2))
            if aux not in pts:
                pts.append(aux)

        return pts
    
    def show(self):
        """Shows the grafic"""
        plt.show()
        
    def save(self, out_img):
        """Save grafic on given path"""
        plt.savefig(out_img)
        
    def clear(self):
        """Clear the instanced axe of pyplot (doesn't work)"""
        self.ax.clear()
        
    def close(self):
        "Closes the instanced figure of pyplot"
        plt.close(self.fig)


# test = bov_plot()
# t = test.rand_line(-10, -9, 9, 10, 50)
# # test.plot_data(t)
# # test.show()
# # test.clear()
# i, o, s = ransac(t, 1, 50)
# test.plot_ransac(i, o, s)
# test.show()
# test.close()