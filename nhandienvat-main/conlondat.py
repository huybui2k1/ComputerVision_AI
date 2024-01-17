# import pyopencl as cl
# import numpy as np

# # Create a PyOpenCL context
# platform = cl.get_platforms()[0]
# device = platform.get_devices()[0]
# context = cl.Context([device])

# # Create a PyOpenCL command queue
# queue = cl.CommandQueue(context)

# # Define an OpenCL kernel
# kernel_source = """
# __kernel void add(__global float* a, __global float* b, __global float* result) {
#     int gid = get_global_id(0);
#     result[gid] = a[gid] + b[gid];
# }
# """

# # Build the OpenCL program
# program = cl.Program(context, kernel_source).build()

# # Create input data
# a = np.array([1, 2, 3, 4], dtype=np.float32)
# b = np.array([5, 6, 7, 8], dtype=np.float32)

# # Allocate OpenCL buffers
# a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
# b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
# result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)

# # Execute the OpenCL kernel
# program.add(queue, a.shape, None, a_buffer, b_buffer, result_buffer)

# # Copy the result from the OpenCL buffer to a NumPy array
# result = np.empty_like(a)
# cl.enqueue_copy(queue, result, result_buffer).wait()

# # Print the result
# print("Result:", result)
import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Get the current GPU device index
    current_device = torch.cuda.current_device()
    
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(current_device)
    
    print(f"Using GPU: {gpu_name} (Device {current_device})")
else:
    print("No GPU available, using CPU.")