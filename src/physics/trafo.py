import odl
import numpy as np

from odl import uniform_discr
from odl.contrib.torch import OperatorModule
from odl.discr import uniform_partition

def simple_trafo(
	im_size, 
	num_angles,
	): 

	ray_trafo = {}

	domain = uniform_discr([-im_size//2, -im_size//2], [im_size//2, im_size//2], (im_size,im_size) , dtype=np.float32)

	geometry = odl.tomo.parallel_beam_geometry(
		domain, num_angles=num_angles
		)
	ray_trafo_op = odl.tomo.RayTransform(
		domain, 
		geometry, 
		impl="astra_cuda"
		)
	ray_trafo_op_module = OperatorModule(ray_trafo_op)
	ray_trafo_adjoint_op_module = OperatorModule(ray_trafo_op.adjoint)

	ray_trafo['ray_trafo'] = ray_trafo_op
	ray_trafo['ray_trafo_module'] = ray_trafo_op_module
	ray_trafo['ray_trafo_adjoint_module'] = ray_trafo_adjoint_op_module
	ray_trafo['fbp_module'] = OperatorModule(odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo_op, frequency_scaling=0.75, filter_type='Hann'))
	return ray_trafo

def limited_angle_trafo(im_size, angular_range):

	ray_trafo = {}

	space = uniform_discr([-im_size//2, -im_size//2], [im_size//2, im_size//2], (im_size,im_size) , dtype=np.float32)


	corners = space.domain.corners()[:, :2]
	rho = np.max(np.linalg.norm(corners, axis=1))

	min_side = min(space.partition.cell_sides[:2])
	omega = np.pi / min_side
	num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1

	det_min_pt = -rho
	det_max_pt = rho
	
	det_shape = num_px_horiz
	num_angles = int(angular_range/180*np.ceil(omega * rho))

	angle_partition = uniform_partition(0, angular_range*np.pi/180, num_angles)
	det_partition = uniform_partition(det_min_pt, det_max_pt, det_shape)

	geometry = odl.tomo.geometry.parallel.Parallel2dGeometry(angle_partition, det_partition)


	ray_trafo_op = odl.tomo.RayTransform(
		space, 
		geometry, 
		impl="astra_cuda"
		)
	ray_trafo_op_module = OperatorModule(ray_trafo_op)
	ray_trafo_adjoint_op_module = OperatorModule(ray_trafo_op.adjoint)

	ray_trafo['ray_trafo'] = ray_trafo_op
	ray_trafo['ray_trafo_module'] = ray_trafo_op_module
	ray_trafo['ray_trafo_adjoint_module'] = ray_trafo_adjoint_op_module
	ray_trafo['fbp_module'] = OperatorModule(odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo_op, frequency_scaling=0.75, filter_type='Hann'))
	return ray_trafo
 