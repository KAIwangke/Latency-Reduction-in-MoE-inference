Mixture-of-Experts (MoE) models have emerged as an effective approach for building large and powerful AI models in an efficient manner. However, the efficiency of distributed MoE inference remains a bottleneck due to communication overhead from the all-to-all operation and load imbalances from skewed expert popularity. This project aims to address these challenges by integrating dynamic gating, quantization, and expert scheduling to reduce latency during MoE inference. We implement these techniques by modifying the open-source FastMoE framework and evaluate their impact across different datasets using V100 GPUs. Our dynamic gating and scheduler implementation decreases runtime by around 1-3% compared to baseline FastMoE. Quantizing to FP8 precision further compresses the model size. Compared to the baseline, the quantized implementation achieves up to 47% lower inference latency on 4 GPUs and 43% lower latency on 2 GPUs by combining these techniques. Overall, our system demonstrates the potential of these optimizations to enhance the deployability of large MoE models.


Compile the package
```
python3 setup.py bdist_wheel;pip install dist/fastmoe-1.1.0-cp310-cp310-linux_x86_64.whl --force-reinstall
```


run with the Transformer-XL

```
bash scripts/run_enwik8_base_moe.sh train
```

run with the Megatron-LM
```
cd fastMoE/examples/megatron/Megatron-LM-2.5/;bash pretrain_gpt_distributed_with_mp.sh
```


@software{FastMoE,
  author = {Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, Jie Tang},
  doi = {10.5281/zenodo.1234},
  month = {12},
  title = {{FastMoE: A Fast Mixture-of-Expert Training System}},
  url = {[FastMoE](https://github.com/laekov/fastmoe)},
  version = {1.1.0},
  year = {2024}
}
