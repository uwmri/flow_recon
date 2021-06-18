
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['build_rotation', 'RigidRegistration']

def build_rotation(theta, phi, psi):
    rot = np.zeros((3,3), dtype=np.float32)
    rot[0, 0] = np.cos(theta) * np.cos(psi)
    rot[0, 1] = -np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(theta) * np.cos(psi)
    rot[0, 2] = np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)

    rot[1, 0] = np.cos(theta) * np.sin(psi)
    rot[1, 1] = np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)
    rot[1, 2] = -np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi)

    rot[2, 0] = -np.sin(theta)
    rot[2, 1] = np.sin(phi) * np.cos(theta)
    rot[2, 2] = np.cos(phi) * np.cos(theta)

    return rot


class RigidRegistration(nn.Module):
    def __init__(self):
        super(RigidRegistration, self).__init__()

        # Number of output of the localization network (Expected image is frames, number of features)
        self.phi = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.theta = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.psi = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.tx = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.ty = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))
        self.tz = torch.nn.Parameter(torch.tensor([0.0]).view(1,1))

    def forward(self, images):

        # R_x = torch.Tensor(
        #     [[1, 0, 0], [0, torch.cos(self.phi_x), -torch.sin(self.phi_x)], [0, torch.sin(self.phi_x), torch.cos(self.phi_x)]])
        # R_y = torch.Tensor(
        #     [[torch.cos(self.phi_y), 0, torch.sin(self.phi_y)], [0, 1, 0], [-torch.sin(self.phi_y), 0, torch.cos(self.phi_y)]])
        # R_z = torch.Tensor(
        #     [[torch.cos(self.phi_z), -torch.sin(self.phi_z), 0], [torch.sin(self.phi_z), torch.cos(self.phi_z), 0], [0, 0, 1]])
        #
        # matrix = torch.mm(torch.mm(R_z, R_y), R_x)
        rot = torch.zeros(3, 4, dtype=images.dtype, device=images.device)
        rot[0, 0] = torch.cos(self.theta) * torch.cos(self.psi)
        rot[0, 1] = -torch.cos(self.phi) * torch.sin(self.psi) + torch.sin(self.phi) * torch.sin(self.theta) * torch.cos(self.psi)
        rot[0, 2] =  torch.sin(self.phi) * torch.sin(self.psi) + torch.cos(self.phi) * torch.sin( self.theta) * torch.cos(self.psi)

        rot[1, 0] = torch.cos(self.theta)*torch.sin(self.psi)
        rot[1, 1] = torch.cos(self.phi) * torch.cos(self.psi) + torch.sin(self.phi) * torch.sin( self.theta) * torch.sin(self.psi)
        rot[1, 2] = -torch.sin(self.phi) * torch.cos(self.psi) + torch.cos(self.phi) * torch.sin(self.theta) * torch.sin( self.psi)

        rot[2, 0] = -torch.sin(self.theta)
        rot[2, 1] = torch.sin( self.phi) * torch.cos( self.theta)
        rot[2, 2] = torch.cos( self.phi ) * torch.cos( self.theta)

        #print(rot)

        rot[0, 3] = self.tx
        rot[1, 3] = self.ty
        rot[2, 3] = self.tz

        # reshape into (Nbatch*Nframes)x2x3 affine matrix
        theta = rot.view(-1, 3, 4)
        #print(theta)

        images = images.view(-1,1,images.shape[-3], images.shape[-2], images.shape[-1])

        # Create affine grid from affine transform
        # affine grid uses matrices from -1 to 1 along each dimension
        grid = F.affine_grid(theta, images.size(), align_corners=False)

        # Sample the data on the grid
        registered = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)



         # print(raw_theta)
        return registered