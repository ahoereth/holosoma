import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.nn.functional as F

class InfiniteFractalPerlin3D(object):
    def __init__(self, shape, resolutions, periods, factors, batch_size=1, generator=torch.random.default_generator):
        # shape: (height, width)
        self.shape = shape
        self.batch_size = batch_size
        self.resolutions = resolutions # List of (res_h, res_w)
        self.periods = periods         # List of temporal periods (frames per cycle)
        self.factors = factors
        self.generator = generator
        self.device = generator.device

        # Precompute spatial grids
        self.grid_shapes = [(shape[0]//res[0], shape[1]//res[1]) for res in resolutions]

        self.linys = [torch.linspace(0,1,gs[0],device=self.device) for gs in self.grid_shapes]
        self.linxs = [torch.linspace(0,1,gs[1],device=self.device) for gs in self.grid_shapes]

        self.masks = []
        for ly, lx in zip(self.linys, self.linxs):
            my = self.fade(ly)
            mx = self.fade(lx)
            masks_octave = {}
            for j in range(2):
                for k in range(2):
                    wy = my if j==1 else torch.flip(my, [0])
                    wx = mx if k==1 else torch.flip(mx, [0])
                    masks_octave[(j,k)] = wy[:,None] * wx[None,:]
            self.masks.append(masks_octave)

        self.gradient_cache = [{} for _ in resolutions]
        self.frame_idx = 0

    def fade(self, t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def get_gradients(self, octave, z_idx):
        cache = self.gradient_cache[octave]
        if z_idx in cache:
            return cache[z_idx]

        # Clean up old cache (keep only recent)
        keys = list(cache.keys())
        for k in keys:
            if k < z_idx - 1: # Keep z_idx-1 just in case, but mainly we move forward
                del cache[k]

        # Generate new gradients
        # Use a deterministic seed based on octave and z_idx
        # We combine them into a single integer seed
        seed = (hash((octave, z_idx)) % (2**31 - 1))
        g_gen = torch.Generator(device=self.device)
        g_gen.manual_seed(seed)

        res = self.resolutions[octave]
        # Shape: (batch_size, ResH+2, ResW+2, 3)
        gradients = torch.randn((self.batch_size, res[0]+2, res[1]+2, 3), device=self.device, generator=g_gen)
        gradients = gradients / (torch.norm(gradients, dim=-1, keepdim=True) + 1e-8)

        cache[z_idx] = gradients
        return gradients

    def generate_frame(self):
        frame_idx = self.frame_idx
        self.frame_idx += 1
        noise = torch.zeros((self.batch_size, self.shape[0], self.shape[1]), device=self.device)

        for octave, factor in enumerate(self.factors):
            period = self.periods[octave]
            z_val = frame_idx / period
            z_idx = int(math.floor(z_val))
            z_frac = z_val - z_idx

            # Get gradients for the two temporal slices
            g0 = self.get_gradients(octave, z_idx)     # (B, ResH+2, ResW+2, 3)
            g1 = self.get_gradients(octave, z_idx + 1) # (B, ResH+2, ResW+2, 3)

            # Precomputed spatial linear coordinates
            lin_y = self.linys[octave]
            lin_x = self.linxs[octave]

            # Temporal weights
            w_z1 = self.fade(torch.tensor(z_frac, device=self.device))
            w_z0 = 1.0 - w_z1

            octave_noise = 0

            for i in range(2): # Temporal corners: 0 (current), 1 (next)
                current_g = g1 if i==1 else g0
                current_w_z = w_z1 if i==1 else w_z0
                dz = z_frac - i

                # Extract components
                gz = current_g[..., 0]
                gy = current_g[..., 1]
                gx = current_g[..., 2]

                # Compute contributions
                # rx = gx * lin_x
                # ry = gy * lin_y
                # offset = gz * dz

                # Expand dimensions for broadcasting
                # gx: (B, ResH+2, ResW+2)
                # lin_x: (GS_X)
                rx = gx[..., None] * lin_x # (B, ResH+2, ResW+2, GS_X)
                ry = gy[..., None] * lin_y # (B, ResH+2, ResW+2, GS_Y)

                # Permute to (B, ResH+2, ResW+2, GS_Y, GS_X)
                prx = rx[..., None, :] # (..., 1, GS_X)
                pry = ry[..., :, None] # (..., GS_Y, 1)

                nrx = -torch.flip(prx, dims=[-1])
                nry = -torch.flip(pry, dims=[-2])

                offset = gz * dz
                offset = offset[..., None, None] # (B, ResH+2, ResW+2, 1, 1)

                for j in range(2): # Y corners
                    for k in range(2): # X corners
                        term_x = prx if k==0 else nrx
                        term_y = pry if j==0 else nry

                        term = offset + term_y + term_x

                        # Slicing to get the grid
                        sl_y = slice(None, -1) if j==0 else slice(1, None)
                        sl_x = slice(None, -1) if k==0 else slice(1, None)

                        term_sliced = term[:, sl_y, sl_x, :, :]

                        # Apply mask
                        mask = self.masks[octave][(j,k)] # (GS_Y, GS_X)

                        # Accumulate
                        octave_noise += current_w_z * mask * term_sliced

            # Reshape and tile
            # octave_noise shape: (B, ResH+1, ResW+1, GS_Y, GS_X)
            # We want (B, (ResH+1)*GS_Y, (ResW+1)*GS_X)
            res_h, res_w = self.resolutions[octave]
            gs_y, gs_x = self.grid_shapes[octave]

            octave_noise = octave_noise.permute(0, 1, 3, 2, 4).reshape(self.batch_size, (res_h+1)*gs_y, (res_w+1)*gs_x)

            # Crop to target shape
            octave_noise = octave_noise[:, :self.shape[0], :self.shape[1]]

            noise += factor * octave_noise

        return noise
