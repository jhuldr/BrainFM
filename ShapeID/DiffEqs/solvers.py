import abc
import torch
from ShapeID.DiffEqs.misc import _assert_increasing, _handle_unused_kwargs

def set_BC_2D(X, BCs): # X: (n_batch, spatial_size); BCs: (batch, 4, BC_shape, data_dim)
    BC_size = BCs.size(2)
    X[:, : BC_size] = BCs[:, 0]
    X[:, - BC_size :] = BCs[:, 1]
    X[:, :, : BC_size] = BCs[:, 2].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    X[:, :, - BC_size :] = BCs[:, 3].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    del BCs
    return X
def set_BC_3D(X, BCs): # X: (n_batch, spatial_size); BCs: (batch, 6, BC_shape, data_dim, dta_dim)
    BC_size = BCs.size(2)
    X[:, : BC_size] = BCs[:, 0]
    X[:, - BC_size :] = BCs[:, 1]
    X[:, :, : BC_size] = BCs[:, 2].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, - BC_size :] = BCs[:, 3].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, :, : BC_size] = BCs[:, 4].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    X[:, :, :, - BC_size :] = BCs[:, 5].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    del BCs
    return X

''' X[t] = X[t] + dBC[t] (dBC[t] = BC[t+1] - BC[t]) '''
def add_dBC_2D(X, dBCs): # X: (n_batch, spatial_size); BCs: (batch, 4, BC_shape, data_dim)
    BC_size = dBCs.size(2)
    X[:, : BC_size] += dBCs[:, 0]
    X[:, - BC_size :] += dBCs[:, 1]
    X[:, :, : BC_size] += dBCs[:, 2].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    X[:, :, - BC_size :] += dBCs[:, 3].permute(0, 2, 1) # (batch, BC_shape, r) -> (batch, r, BC_shape)
    del dBCs
    return X
def add_dBC_3D(X, dBCs): # X: (n_batch, spatial_size); BCs: (batch, 6, BC_shape, data_dim, dta_dim)
    BC_size = dBCs.size(2)
    X[:, : BC_size] += dBCs[:, 0]
    X[:, - BC_size :] += dBCs[:, 1]
    X[:, :, : BC_size] += dBCs[:, 2].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, - BC_size :] += dBCs[:, 3].permute(0, 2, 1, 3) # (batch, BC_shape, s, c) -> (batch, s, BC_shape, c)
    X[:, :, :, : BC_size] += dBCs[:, 4].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    X[:, :, :, - BC_size :] += dBCs[:, 5].permute(0, 2, 3, 1) # (batch, BC_shape, s, r) -> (batch, s, r, BC_shape)
    del dBCs
    return X

class AdaptiveStepsizeODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, atol, rtol, options= None):
        
       # _handle_unused_kwargs(self, options)
        #del options
        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol

    def before_integrate(self, t):
        pass

    @abc.abstractmethod
    def advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        _assert_increasing(t)
        solution = [self.y0]
        t = t.to(self.y0[0].device, torch.float64)
        self.before_integrate(t)
        for i in range(1, len(t)):
            y = self.advance(t[i])
            solution.append(y)
        '''if self.contours is not None: # contours: (n_batch, nT, 4 / 6, BC_size, c)
            if self.adjoint:
                for i in range(1, len(t)):
                    ys = list(self.advance(t[i])) # tuple: (y0, **back_grad) -> y0: (n_batch, spatial_shape)
                    #print(len(t))
                    #print(ys[0].size())
                    #print(self.contours.size())
                    ys[0] = self.set_BC(ys[0], self.contours[:, i]) # (n_batch, 4 / 6, BC_size, c)
                    solution.append(tuple(ys))
            else:
                for i in range(1, len(t)):
                    y = torch.stack(self.advance(t[i])) # y: (n_batch, 1, spatial_shape) 
                    y = self.set_BC(y[:, 0], self.contours[:, i]).unsqueeze(1) 
                    solution.append(tuple(y))
        elif self.dcontours is not None: # dcontours: (n_batch, nT, 4 / 6, BC_size, c)
            if self.adjoint:
                for i in range(1, len(t)):
                    ys = list(self.advance(t[i])) # ys - tuple: (y0, **back_grad) -> y0: (n_batch, spatial_shape)
                    ys[0] = self.add_dBC(ys[0], self.dcontours[:, i]) # (n_batch, 4 / 6, BC_size, c)
                    solution.append(tuple(ys))
            else:
                for i in range(1, len(t)):
                    y = torch.stack(self.advance(t[i])) # (n_batch, 1, spatial_shape)
                    y = self.add_dBC(y[:, 0], self.dcontours[:, i]).unsqueeze(1)
                    solution.append(tuple(y))
        else:
            for i in range(1, len(t)):
                y = self.advance(t[i])
                solution.append(y)'''
        return tuple(map(torch.stack, tuple(zip(*solution))))


class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, step_size=None, grid_constructor=None, atol=None, rtol=None, dt=None, options = None):
        '''if 'dirichlet' in options.BC or 'cauchy' in options.BC and options.contours is not None:
            self.contours = options.contours  # (n_batch, nT, 4 / 6, BC_size, sub_spatial_shape)
            self.BC_size = self.contours.size(3)
            self.set_BC = set_BC_2D if self.contours.size(2) == 4 else set_BC_3D
        else:
            self.contours = None
        if 'source' in options.BC and options.dcontours is not None:
            self.dcontours = options.dcontours  # (n_batch, nT, 4 / 6, BC_size, sub_spatial_shape)
            self.BC_size = self.dcontours.size(3)
            self.add_dBC = add_dBC_2D if self.dcontours.size(2) == 4 else add_dBC_3D
        else:
            self.dcontours = None'''
        #self.adjoint = options.adjoint
        #options.pop('rtol', None)
        #options.pop('atol', None)
        #_handle_unused_kwargs(self, options)
        #del options

        self.func = func
        self.y0 = y0

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda f, y0, t: t # Same time step as time interval
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]
            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        _assert_increasing(t)
        t = t.type_as(self.y0[0]) # (n_time, )
        time_grid = self.grid_constructor(self.func, self.y0, t)
        #print('time_grid:', time_grid.size())
        #print('t:', t.size())
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])

        solution = [self.y0]

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self.step_func(self.func, t0, t1 - t0, y0)
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))
            y0 = y1
            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1
            '''if self.contours is not None:
                if self.adjoint:
                    for i in range(1, len(t)):
                        ys = list(self._linear_interp(t0, t1, y0, y1, t[j])) # tuple: (y0, **back_grad) -> y0: (n_batch, spatial_shape)
                        ys[0] = self.set_BC(ys[0], self.contours[:, i]) # (n_batch, 4 / 6, BC_size, c)
                        solution.append(tuple(ys))
                        j += 1
                else:
                    while j < len(t) and t1 >= t[j]:
                        y = torch.stack(self._linear_interp(t0, t1, y0, y1, t[j])) # (n_batch, 1, spatial_shape)
                        y = self.set_BC(y[:, 0], self.contours[:, j]).unsqueeze(1) 
                        solution.append(tuple(y))
                        j += 1
            elif self.dcontours is not None:
                if self.adjoint:
                    for i in range(1, len(t)):
                        ys = list(self._linear_interp(t0, t1, y0, y1, t[j])) # tuple: (y0, **back_grad) -> y0: (n_batch, spatial_shape)
                        ys[0] = self.add_dBC(ys[0], self.dcontours[:, j]) # (n_batch, 4 / 6, BC_size, c)
                        solution.append(tuple(ys))
                else:
                    while j < len(t) and t1 >= t[j]:
                        y = torch.stack(self._linear_interp(t0, t1, y0, y1, t[j])) # (n_batch, 1, spatial_shape)
                        y = self.add_dBC(y[:, 0], self.dcontours[:, j]).unsqueeze(1)
                        solution.append(tuple(y))
                        j += 1
            else:
                while j < len(t) and t1 >= t[j]:
                    solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                    j += 1'''
        return tuple(map(torch.stack, tuple(zip(*solution)))) # (batch, time)

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))
