import numpy as np
from scipy.integrate import odeint, solve_ivp


def get_spring_data(n_ics):
    t, x, dx, ddx, z = generate_spring_data(n_ics)
    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics * t.size, -1))
    data['dx'] = dx.reshape((n_ics * t.size, -1))
    data['ddx'] = ddx.reshape((n_ics * t.size, -1))
    data['z'] = z.reshape((n_ics * t.size, -1))[:, 0:1]
    data['dz'] = z.reshape((n_ics * t.size, -1))[:, 1:2]

    return data


def generate_spring_data(n_ics, c=0.33, m=0.5, k=1.):
    f = lambda z, t: [z[1], -c * z[1] - (k / m) * z[0]]
    t = np.arange(0, 20, .02)

    z = np.zeros((n_ics, t.size, 2))
    dz = np.zeros(z.shape)

    z1range = np.array([0., 1.])
    z2range = np.array([-0.1, 0.1])
    i = 0
    while i < n_ics:
        z0 = np.array([(z1range[1] - z1range[0]) * np.random.rand() + z1range[0],
                       (z2range[1] - z2range[0]) * np.random.rand() + z2range[0]])
        # what is this for? it was there in the pendulum case
        #if np.abs(z0[1] ** 2 / 2. - np.cos(z0[0])) > .99:
        #    continue
        z[i] = odeint(f, z0, t)
        # z[i] = solve_ivp(f, t_span=(t[0], t[-1]), y0=z0, t_eval=t).y.T
        # print(z[0].shape)
        dz[i] = np.array([f(z[i, j], t[j]) for j in range(len(t))])
        i += 1

    x, dx, ddx = spring_to_movie(z, dz)

    ### COMMENTS NOT UPDATED ###
    # n = 51
    # xx,yy = np.meshgrid(np.linspace(-1.5,1.5,n),np.linspace(1.5,-1.5,n))
    # create_image = lambda theta : np.exp(-((xx-np.cos(theta-np.pi/2))**2 + (yy-np.sin(theta-np.pi/2))**2)/.05)
    # argument_derivative = lambda theta,dtheta : -1/.05*(2*(xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta \
    #                                                   + 2*(yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta)
    # argument_derivative2 = lambda theta,dtheta,ddtheta : -2/.05*((np.sin(theta-np.pi/2))*np.sin(theta-np.pi/2)*dtheta**2 \
    #                                                            + (xx - np.cos(theta-np.pi/2))*np.cos(theta-np.pi/2)*dtheta**2 \
    #                                                            + (xx - np.cos(theta-np.pi/2))*np.sin(theta-np.pi/2)*ddtheta \
    #                                                            + (-np.cos(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*dtheta**2 \
    #                                                            + (yy - np.sin(theta-np.pi/2))*(np.sin(theta-np.pi/2))*dtheta**2 \
    #                                                            + (yy - np.sin(theta-np.pi/2))*(-np.cos(theta-np.pi/2))*ddtheta)

    # x = np.zeros((n_ics, t.size, n, n))
    # dx = np.zeros((n_ics, t.size, n, n))
    # ddx = np.zeros((n_ics, t.size, n, n))
    # for i in range(n_ics):
    #     for j in range(t.size):
    #         z[i,j,0] = wrap_to_pi(z[i,j,0])
    #         x[i,j] = create_image(z[i,j,0])
    #         dx[i,j] = (create_image(z[i,j,0])*argument_derivative(z[i,j,0], dz[i,j,0]))
    #         ddx[i,j] = create_image(z[i,j,0])*((argument_derivative(z[i,j,0], dz[i,j,0]))**2 \
    #                         + argument_derivative2(z[i,j,0], dz[i,j,0], dz[i,j,1]))

    return t, x, dx, ddx, z

# putting together nate's "spring_preliminary" and the pendulum example
def spring_to_movie(z, dz, n = 30):
    n_ics = z.shape[0]
    n_samples = z.shape[1]

    y1, y2 = np.meshgrid(np.linspace(-1.5, 1.5, n), np.linspace(1.5, -1.5, n))

    # defining a guassian over the image centered where our point mass is
    create_image = lambda x: np.exp(-((y1 - x) ** 2 + (y2 - 0) ** 2) / .05)
    d_create_image = lambda x, dx: -1 / .05 * create_image(x) * 2 * (y1 - x) * dx
    dd_create_image = lambda x, dx, ddx: (2 / .05) * (d_create_image(x, dx) * (y1 - x) * dx + create_image(x) * (- dx * dx + (y1 - x) * ddx))

    x = np.zeros((n_ics, n_samples, n, n))
    dx = np.zeros((n_ics, n_samples, n, n))
    ddx = np.zeros((n_ics, n_samples, n, n))
    for i in range(n_ics):
        for j in range(n_samples):
            #z[i, j, 0] = wrap_to_pi(z[i, j, 0]) removed from pendulum
            x[i, j] = create_image(z[i, j, 0])
            dx[i, j] = d_create_image(z[i, j, 0], dz[i, j, 0])
            ddx[i, j] = dd_create_image(z[i, j, 0], dz[i, j, 0], dz[i, j, 1])
    return x, dx, ddx

