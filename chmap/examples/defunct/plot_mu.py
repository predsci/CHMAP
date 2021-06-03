# importing the required module
import matplotlib.pyplot as plt
import numpy

# create outline circle
out_r = 1
out_ang = numpy.array(range(101), dtype='f')/100 * 2*numpy.pi
out_x = out_r*numpy.sin(out_ang)
out_y = out_r*numpy.cos(out_ang)

# create equi-spaced mu circles
n_radii = 5
mu_radii = (numpy.array(range(n_radii), dtype='f')+0.8)/n_radii
# theta=0 at center of image
theta_r = numpy.arccos(mu_radii)
image_r = numpy.sin(theta_r)


# create example pixel outline
pix_x = [0.5, 0.5, 0.45, 0.45, 0.5]
pix_y = [0.5, 0.45, 0.45, 0.5, 0.5]

fig1 = plt.figure(1)
ax1  = fig1.gca()

# plotting the points
ax1.plot(out_x, out_y, label="outline", color="black")
ax1.plot(pix_x, pix_y, label="pixel", color="red")

# plot equi-mu circles
for temp_r in image_r:
    temp_x = temp_r*numpy.sin(out_ang)
    temp_y = temp_r*numpy.cos(out_ang)
    ax1.plot(temp_x, temp_y, color="grey")

# naming the x axis
plt.xlabel('image x - axis')
# naming the y axis
plt.ylabel('image y - axis')
# make legend
plt.legend()
# equal x and y scales
ax1.set_aspect('equal', adjustable='box')

# giving a title to my graph
plt.title('Test Image')



# convert outline to map
map_out_y = out_y
out_helio_z = out_y
out_helio_y = out_x
out_helio_x = 0

out_helio_theta = numpy.arccos(out_helio_z)
out_helio_phi = numpy.sign(out_helio_y) * numpy.pi/2

map_out_x = out_helio_phi

# convert pixel to map
map_pix_y = pix_y
pix_helio_z = numpy.array(pix_y)
pix_helio_y = numpy.array(pix_x)
pix_helio_x = numpy.sqrt(1 - pix_helio_y**2 - pix_helio_z**2)

pix_helio_phi = numpy.arctan(pix_helio_y/pix_helio_x)

# plot
fig2 = plt.figure(2)
ax2  = fig2.gca()

ax2.plot(map_out_x, map_out_y, color="black")
ax2.plot(map_pix_y, pix_helio_phi, color="red")

# plot equi-mu circles
for temp_r in image_r:
    temp_x = temp_r*numpy.sin(out_ang)
    temp_y = temp_r*numpy.cos(out_ang)

    map_temp_y = temp_y
    temp_helio_z = numpy.array(temp_y)
    temp_helio_y = numpy.array(temp_x)
    temp_helio_x = numpy.sqrt(1 - temp_helio_y**2 - temp_helio_z**2)

    temp_helio_phi = numpy.arctan(temp_helio_y/temp_helio_x)

    ax2.plot(temp_helio_phi, map_temp_y, color="grey")

plt.xlabel(r'$\Phi$')
plt.ylabel(r'$\sin(\zeta)$')
plt.title("Corresponding Map")
# equal x and y scales
ax2.set_aspect('equal', adjustable='box')

# function to show the plot
plt.show()


# calc concentric annulli area for equi-distant mus
n_mu = int(numpy.floor((1-.14037)/.06))
mu_series = 1 - (numpy.array(range(n_mu))+1)*.06
theta_series = numpy.arccos(mu_series)
rad_series = numpy.sin(theta_series)
area_series = rad_series**2 * numpy.pi
annuli_series = numpy.diff(area_series)