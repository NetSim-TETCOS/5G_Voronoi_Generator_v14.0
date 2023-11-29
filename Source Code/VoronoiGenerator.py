import os 
import sys
import re 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d

node_x = []
node_y = []
device_id = []

def config_reader():
    device_name=""
    l_flag=0
    #if not (os.path.isfile('Configuration.netsim')):
    #    print("Error: Configuration.netsim file missing in path: "+sys.argv[1])
    #    sys.exit()
    for i, line in enumerate(open('configuration.netsim')):
        try:
            if l_flag==0:
                found=re.search("<DEVICE KEY=\"Macrocell_Omni_gNB\" DEVICE_NAME=\"(.+?)\" DEVICE_ID=\"(.+?)\" TYPE=\"GNB\" INTERFACE_COUNT=\"(.+?)\" DEVICE_ICON=\"(.+?)\">",line).group(1,2,3)
                l_flag=1    
                device_id.append(found[1])
            else:
                found=re.search("<POS_3D X_OR_LON=\"(.+?)\" Y_OR_LAT=\"(.+?)\" Z=\"(.+?)\" COORDINATE_SYSTEM=\"Cartesian\" ICON_ROTATION=\"(.+?)\" />",line).group(1,2)
                if l_flag==1:
                    node_x.append(float(found[0]))
                    node_y.append(float(found[1]))
                l_flag=0                               

        except AttributeError:
            pass   

def voronoi_finite_polygons_2d(vor, radius=None):
    
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

if len(sys.argv) == 1:
   print('Error: No Arguments Passed\nPass the path of the saved 5G experiment directory as argument to get the Voronoi Plot.')
   sys.exit()
elif len(sys.argv) >= 2:
    if not(os.path.exists(sys.argv[1])):
        print('Error: Invalid Experiment path. Pass the path of the saved 5G experiment directory as argument to get the Voronoi Plot.')
        sys.exit() 

os.chdir(sys.argv[1])
tracepath = sys.argv[1]+'\Configuration.netsim'

if not (os.path.isfile(tracepath)):
    print("Error: Configuration.netsim file missing in path: "+sys.argv[1])
    sys.exit()

config_reader()
points = np.column_stack((node_x, node_y))  
vor = Voronoi(points, qhull_options='QJ')
fig, ax = plt.subplots(figsize=(8, 8),num="Voronoi Tesselation")
# Plot Voronoi regions with colors
voronoi_plot_2d(vor, ax=ax.twiny(), show_vertices=False, line_colors='blue', line_width=0.5, point_size=0)
# plot
regions, vertices = voronoi_finite_polygons_2d(vor)
ax.xaxis.set_visible(False)
# colorize
for region in regions:
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0)

for i in range(0,np.size(node_x)):
    plt.text(node_x[i]+10,node_y[i]+25,device_id[i],fontsize = 8)


plt.scatter(points[:,0], points[:,1],color = 'green',s=6)
plt.xlim(0, vor.max_bound[0] + 600)
plt.ylim(0, vor.max_bound[1] + 600)
plt.title("Voronoi Tesselation")
ax.invert_yaxis()
plt.show()