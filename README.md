recompile


```
python3 setup.py bdist_wheel;pip install dist/fastmoe-1.1.0-cp310-cp310-linux_x86_64.whl --force-reinstall
```


run


```
bash scripts/run_enwik8_base_moe.sh train
```







scheduler



Expert popularity estimation:

Add a new module to estimate the expert popularity distribution for each incoming batch of inference requests.
Utilize the token-level expert selection patterns across MoE layers, as observed in Lina.
During the profiling stage, collect the expert selection results for each token when the load balancing loss is minimized and stable.
Group tokens based on their selected experts in the past few layers to form a unique sample path.
Compute the expert popularity distribution for the next layer based on the sample paths.
Implement this estimation logic in a new file, e.g., expert_popularity_estimator.cu.


Two-phase scheduling:

Implement the two-phase scheduling approach used in Lina to dynamically assign experts to GPUs based on the estimated popularity.
In phase one, use the estimated popularity distribution to determine the number of GPUs to allocate for each expert. Assign more GPUs to popular experts and pack less popular ones onto fewer GPUs.
Modify the expert assignment logic in FastMoE to incorporate this dynamic scheduling. This may involve changes to the fmoe_cuda_expert_count_impl function in local_exchange.cuh.
In phase two, after the actual expert selections are made by the gating network, compare the actual popularity with the estimated one. If there is a significant deviation, fine-tune the expert-to-GPU assignment accordingly.
Implement the two-phase scheduling logic in a new file, e.g., dynamic_expert_scheduler.cu.


Load balancing across GPUs:

Modify the all-to-all communication primitives in FastMoE to support unequal split across GPUs, as done in Lina.
Update the fmoe_cuda_global_scatter_impl and fmoe_cuda_global_gather_impl functions in global_exchange.cu to handle the case where each GPU may have a different number of assigned tokens based on the expert popularity.
Ensure that the workload is balanced across GPUs by considering the number of tokens assigned to each expert replica.


Expert packing and unpacking:

Implement the logic to pack and unpack experts on GPUs based on the dynamic scheduling decisions.
Modify the expert forward and backward pass in parallel_linear.cuh to handle the case where multiple experts may be assigned to the same GPU.
Optimize the expert computation by utilizing stream parallelism and overlapping communication with computation when multiple experts are packed on the same GPU.


Integration with FastMoE:

Integrate the new modules for expert popularity estimation and dynamic scheduling into the FastMoE codebase.
Modify the inference pipeline to incorporate the two-phase scheduling approach.
Update the necessary data structures and interfaces to support dynamic expert-to-GPU assignment.
Ensure proper synchronization and communication between the scheduler and the rest of the FastMoE components.


Testing and optimization:

Develop comprehensive unit tests to verify the correctness of the expert popularity estimation and dynamic scheduling modules.
Conduct performance profiling and analysis to measure the effectiveness of the dynamic scheduling in terms of load balancing and inference latency.
Fine-tune the hyperparameters, such as the number of historical layers considered for popularity estimation and the threshold for fine-tuning in phase two.
Optimize the implementation for efficiency, considering factors like memory usage, communication overhead, and computation-communication overlap.


