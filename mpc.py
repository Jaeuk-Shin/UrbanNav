
import numpy as np
import casadi as ca
import time


class MPC:
    """
    A simple MPC implementation for tracking waypoints; assume a unicycle model
    """
    def __init__(self, ts, dt, ulb, uub, max_wall_time=0.1):
        self.ts = ts
        self.dt = dt
        self.ulb = ulb
        self.uub = uub

        self.xlb = np.array([-10., -10., -3. * np.pi])
        self.xub = np.array([10., 10., 3. * np.pi])

        # state & ctrl dim.
        self.dx = 3  # (x, y, theta)
        self.du = 2  # (v, w)

        # self.max_wall_time = max_wall_time  # in seconds
        self.solver = self.build_solver(max_wall_time)
        
        # for warm start
        self.X = np.zeros((self.ts+1, 3))
        self.U = np.zeros((self.ts, 2))




    def build_solver(self, max_wall_time):
        ts = self.ts

        dx, du = self.dx, self.du

        # optimization variables
        # the following indexing used here:
        # (i, t) -> i + d_x * t
        # where i: state idx / t: time
        x = ca.MX.sym('x', dx*(ts+1))  # states
        u = ca.MX.sym('u', du*ts)      # controls

        p = ca.MX.sym('p', dx+ts*2+ts)    # initial states + waypoints + state cost weights

        def unicycle_dynamics(x1, u1):
            # dynamics function for the unicycle model
            v, w = u1[0], u1[1]
            th = x1[2]
            dx1 = ca.vertcat(v * ca.cos(th), v * ca.sin(th), w)
            return x1 + dx1 * self.dt

        # constraints & obj.
        g = []  # list of constraints
        self.lbg = []  # lower bounds
        self.ubg = []  # upper bounds
        objective = 0.

        # init state constraints
        x0 = x[:dx]
        g.append(x0 - p[:dx])
        self.lbg += [0.] * dx
        self.ubg += [0.] * dx

        xlb_rep = np.tile(self.xlb, reps=ts+1)
        xub_rep = np.tile(self.xub, reps=ts+1)
        ulb_rep = np.tile(self.ulb, reps=ts)
        uub_rep = np.tile(self.uub, reps=ts)

        self.idx_split = dx * (ts + 1)     # before -> state / after -> control
        # lower bounds
        self.lbx = -ca.inf * ca.DM.ones(dx*(ts+1)+du*ts)
        self.lbx[:self.idx_split] = xlb_rep
        self.lbx[self.idx_split:] = ulb_rep
        # upper bounds
        self.ubx = ca.inf * ca.DM.ones(dx*(ts+1)+du*ts)
        self.ubx[:self.idx_split] = xub_rep
        self.ubx[self.idx_split:] = uub_rep

        for t in range(ts):
            xt_begin = dx * t
            ut_begin = du * t
            xt_next_begin = xt_begin + dx
            xt = x[xt_begin: xt_begin+dx]
            ut = u[ut_begin: ut_begin+du]

            # dynamics constraints (eq.)
            xt_next = x[xt_next_begin: xt_next_begin+dx]
            g.append(xt_next - unicycle_dynamics(xt, ut))
            self.lbg += [0. for _ in range(dx)]
            self.ubg += [0. for _ in range(dx)]

            pos_t_next = x[xt_next_begin: xt_next_begin+2]

            # objective
            # prioritize reaching the first waypoint
            cost_weight = 5. if t <= 0 else 1.
            objective += p[dx+2*ts+t] * ca.sumsqr(pos_t_next - p[dx+2*t:dx+2*(t+1)]) ** 2 + 1e-4 * ca.sumsqr(u) ** 2

        # Flatten constraints
        g = ca.vertcat(*g)

        # define the optimization problem
        opts = {
            'ipopt.print_level': 1, 
            'ipopt.max_wall_time': max_wall_time,
            'print_time': 0, 
            'verbose': True, 
            'jit': True, 
            'compiler': 'shell', 
            'jit_options': {'flags': ['-O3']}
            }
        nlp = {'x': ca.vertcat(x, u), 'f': objective, 'g': g, 'p': p}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        return solver

    def solve(self,
        # geometry: Geometry, r_robot,
        initial_pose,
        waypoints,
        cost_weights,
        ):
        """
        ts: timesteps; [0, ..., ts] denotes the planning horizon
        dt: sampling time
        xlb, xub, ulb, uub: 1-dim. numpy arrays representing the state & ctrl bounds of the ego-robot
        waypoints: numpy array of shape (ts, 2) representing the sequence of N waypoints
        update_frequency: time between consecutive waypoints
        """

        t_wall_begin = time.perf_counter()

        # provide the initial guesses
        x_init = np.reshape(self.X, (-1,))
        u_init = np.reshape(self.U, (-1,))

        x0 = np.concatenate([x_init, u_init])

        # parameters of the optimization problem
        p_input = np.concatenate((initial_pose, np.reshape(waypoints, (-1,)), cost_weights))

        # solve the optimization
        sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p_input)

        x_sol = np.reshape(sol['x'][:self.idx_split], (-1, self.dx))
        u_sol = np.reshape(sol['x'][self.idx_split:], (-1, self.du))

        self.X = np.copy(x_sol)
        self.U = np.copy(u_sol)

        stats = self.solver.stats()

        t_wall = time.perf_counter() - t_wall_begin

        # t_wall = sum(t for key, t in stats.items() if key.startswith('t_wall'))


        stats = {'t_wall': t_wall}


        return x_sol, u_sol, stats