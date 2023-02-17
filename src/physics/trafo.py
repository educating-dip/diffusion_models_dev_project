import odl
import numpy as np

from odl import uniform_discr
from odl.contrib.torch import OperatorModule

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
