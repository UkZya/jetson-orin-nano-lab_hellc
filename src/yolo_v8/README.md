
Speed: 7.3ms preprocess, 154.6ms inference, 20.6ms postprocess

=== Benchmark Result ===
avg latency : 40.91 ms
min latency : 39.30 ms
max latency : 48.23 ms
fps         : 24.45


=== Stage-wise Benchmark Result ===
preprocess   avg=   5.61 ms | min=   5.16 ms | max=   6.31 ms | ratio= 17.50%
inference    avg=  22.41 ms | min=  21.67 ms | max=  22.80 ms | ratio= 69.94%
postprocess  avg=   4.02 ms | min=   3.93 ms | max=   4.17 ms | ratio= 12.56%
total        avg=  32.03 ms | min=  30.81 ms | max=  32.95 ms


=== Forward-only Benchmark Result ===
forward      avg=  28.00 ms | min=  27.65 ms | max=  35.75 ms
fps         : 35.71


=== Module-wise Profiling Result (sorted by total time) ===
module                                                           avg_ms     total_ms    calls
-----------------------------------------------------------------------------------------------
model.0.act                                                       0.165      281.558     1710
model.0.conv                                                      6.007      180.212       30
model.2.cv1.conv                                                  3.090       92.708       30
model.1.conv                                                      2.538       76.140       30
model.22.cv3.0.1.conv                                             1.671       50.124       30
model.2.m.0.cv1.conv                                              1.630       48.907       30
model.22.cv3.0.0.conv                                             1.345       40.341       30
model.4.m.0.cv1.conv                                              1.169       35.063       30
model.0.bn                                                        1.160       34.810       30
model.22.cv2.0.0.conv                                             1.141       34.238       30
model.22.dfl.conv                                                 1.110       33.289       30
model.7.conv                                                      1.076       32.269       30
model.3.conv                                                      1.048       31.453       30
model.4.m.1.cv1.conv                                              1.007       30.201       30
model.4.m.1.cv2.conv                                              1.006       30.168       30
model.4.m.0.cv2.conv                                              1.003       30.102       30
model.22.cv2.0.1.conv                                             1.001       30.035       30
model.15.m.0.cv1.conv                                             0.989       29.684       30
model.15.m.0.cv2.conv                                             0.973       29.191       30
model.22.cv2.1.0.conv                                             0.906       27.167       30
model.5.conv                                                      0.859       25.770       30
model.9.m                                                         0.278       25.019       90
model.22.cv2.2.0.conv                                             0.822       24.653       30
model.22.cv3.1.0.conv                                             0.818       24.534       30
model.22.cv3.1.1.conv                                             0.788       23.648       30
model.2.m.0.cv2.conv                                              0.769       23.057       30
model.22.cv3.2.0.conv                                             0.757       22.702       30
model.8.m.0.cv1.conv                                              0.737       22.102       30
model.2.cv2.conv                                                  0.729       21.865       30
model.19.conv                                                     0.715       21.465       30
model.22.cv3.0.2                                                  0.618       18.542       30
model.16.conv                                                     0.618       18.538       30
model.15.cv1.conv                                                 0.597       17.918       30
model.4.cv2.conv                                                  0.593       17.775       30
model.12.cv1.conv                                                 0.580       17.403       30
model.14                                                          0.544       16.326       30
model.21.m.0.cv1.conv                                             0.542       16.256       30
model.2.cv2.bn                                                    0.534       16.018       30
model.8.m.0.cv2.conv                                              0.533       16.003       30
model.6.m.0.cv1.conv                                              0.533       15.984       30
model.6.cv2.conv                                                  0.530       15.896       30
model.21.m.0.cv2.conv                                             0.528       15.835       30
model.22.cv3.2.1.conv                                             0.525       15.750       30
model.1.bn                                                        0.520       15.587       30
model.2.cv1.bn                                                    0.518       15.555       30
model.15.cv2.conv                                                 0.464       13.932       30
model.4.cv1.conv                                                  0.457       13.701       30
model.22.cv2.0.2                                                  0.419       12.565       30
model.12.cv2.conv                                                 0.406       12.184       30
model.9.cv2.conv                                                  0.401       12.034       30



=== Pretty Op-wise Profiling (Top 15) ===
[INFO] Time domain: CPU
op                                      time(ms)      ratio      calls
------------------------------------------------------------------------
aten::conv2d                             164.916     14.19%       1280
aten::convolution                        159.206     13.70%       1280
aten::_convolution                       149.525     12.86%       1280
aten::batch_norm                         136.808     11.77%       1140
aten::_batch_norm_impl_index             130.964     11.27%       1140
aten::cudnn_convolution                  129.083     11.11%       1280
aten::cudnn_batch_norm                   122.864     10.57%       1140
aten::silu_                               40.121      3.45%       1140
aten::empty                               32.992      2.84%       4580
aten::empty_like                          19.803      1.70%       1160
aten::cat                                 15.859      1.36%        340
aten::chunk                                7.336      0.63%        180
aten::add                                  6.674      0.57%        160
aten::split                                6.491      0.56%        180
aten::view                                 5.973      0.51%       1420


1순위
aten::cudnn_convolution
aten::cudnn_batch_norm
2순위
aten::silu_
3순위
aten::cat
aten::empty
aten::empty_like



python -c "import onnx; m = onnx.load('yolov8n.onnx'); onnx.checker.check_model(m); print('ONNX OK')"

=== ONNX Runtime Benchmark Result ===
total        avg= 385.64 ms | min= 191.41 ms | max= 725.77 ms
preprocess   avg=  12.89 ms | min=   7.16 ms | max=  25.89 ms
inference    avg= 372.75 ms | min= 181.60 ms | max= 708.62 ms
fps         : 2.59


tensorrt build
trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=engines/yolov8n_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --verbose

  tensorrt benchmark
  trtexec \
  --loadEngine=engines/yolov8n_fp16.engine \
  --warmUp=100 \
  --iterations=200 \
  --useCudaGraph

  TRT FP16 latency mean: 6.12 ms
GPU compute time mean: 5.41 ms
Throughput: 183.5 qps