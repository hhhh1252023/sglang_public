# 235b_run.sh
# docker exec -it sglang_perf_b150 bash
pkill -9 python | pkill -9 sglang
pkill -9 sglang
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH

cd /home/chenxu/sglang_ascend_1111/sglang_ascend
export PYTHONPATH=${PWD}/python:$PYTHONPATH
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING


source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
#export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/home/chenxu/sglang_ascend_1111/sglang_ascend/hot_map
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

MODEL_PATH=/mnt/share/weights/Qwen3-235B-A22B-W8A8
# pd传输, IP设置为p节点首节点
export ASCEND_MF_STORE_URL="tcp://141.61.105.141:24667"
# p节点IP
P_IP=('141.61.105.141')
# D节点IP
#D_IP=('141.61.105.144')
D_IP=('141.61.105.143' '141.61.105.144')
#export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"


for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        export HCCL_BUFFSIZE=3000
        export TASK_QUEUE_ENABLE=2
        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        export STREAMS_PER_DEVICE=32
        export ENABLE_ASCEND_MOE_NZ=1
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
#        export ENABLE_PROFILING=1

        # P节点
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode prefill \
        --host ${P_IP[$i]} --port 8000 --disaggregation-bootstrap-port 8995 --trust-remote-code \
        --nnodes 1 --node-rank $i --tp-size 16 --dp-size 8 --mem-fraction-static 0.6 \
        --disable-radix-cache \
        --ep-dispatch-algorithm static --init-expert-location /home/chenxu/sglang_ascend_1111/sglang_ascend/hot_map/expert_distribution_recorder_1763480391.7582676.pt \
        --attention-backend ascend --device npu --quantization w8a8_int8 --disaggregation-transfer-backend ascend \
        --max-running-requests 128 --chunked-prefill-size 114688 --max-prefill-tokens 458880 \
        --disable-overlap-schedule  --enable-dp-attention --tokenizer-worker-num 4 \
        --moe-a2a-backend deepep --deepep-mode normal --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done


for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        export DP_ROUND_ROBIN=1
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=60
        export HCCL_BUFFSIZE=512
        export HCCL_SOCKET_IFNAME=data0.3001
        export GLOO_SOCKET_IFNAME=data0.3001
        export STREAMS_PER_DEVICE=32

#        export ENABLE_ASCEND_MOE_NZ=1
#        export ENABLE_PROFILING=1
        # D节点
        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
        --host ${D_IP[$i]} --port 8001 --trust-remote-code \
        --nnodes 2 --node-rank $i --tp-size 32 --dp-size 16 --mem-fraction-static 0.83 --max-running-requests 960 \
        --attention-backend ascend --device npu --quantization w8a8_int8 --enable-dp-attention \
        --moe-a2a-backend ascend_fuseep --cuda-graph-bs 6 8 12 15 18 20 22 24 30 36 42 48 54 56 58 60 \
        --dist-init-addr 172.27.1.143:5000 \
        --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
        --prefill-round-robin-balance --enable-dp-lm-head --tokenizer-worker-num 4 --dtype bfloat16
        NODE_RANK=$i
        break
    fi
done

# tp2dp16
#for i in "${!D_IP[@]}";
#do
#    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
#    then
#        echo "${D_IP[$i]}"
#        export DP_ROUND_ROBIN=1
#        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
#        export HCCL_BUFFSIZE=512
#        export HCCL_SOCKET_IFNAME=data0.3001
#        export GLOO_SOCKET_IFNAME=data0.3001
##        export HCCL_SOCKET_IFNAME=enp194s0f0
##        export GLOO_SOCKET_IFNAME=enp194s0f0
#
#        export STREAMS_PER_DEVICE=32
#        export SGLANG_USE_GATING_TOPK_FUSED=1
#        export SGLANG_USE_TRITON_SPLIT_RMSNORM=1
#        export SGLANG_USE_ADD_NORM_BIAS_QUANT=1
##        export SGLANG_USE_TRITON_ADD_NORM_BIAS=1
#        export SGLANG_USE_NORM_L1_TRITON=1
##        export ENABLE_PROFILING=1
##        export ENABLE_FUSED_MOE=1
#        # D节点
#        python -m sglang.launch_server --model-path ${MODEL_PATH} --disaggregation-mode decode \
#        --host ${D_IP[$i]} --port 8001 --trust-remote-code \
#        --nnodes 1 --node-rank $i --tp-size 16 --dp-size 8 --mem-fraction-static 0.8 --max-running-requests 144 \
#        --attention-backend ascend --device npu --quantization w8a8_int8 --enable-deepep-moe --enable-dp-attention \
#        --deepep-mode low_latency --cuda-graph-bs 2 4 6 8 12 14 15 18 \
#        --dist-init-addr 172.27.1.143:5000 \
#        --disaggregation-transfer-backend ascend --watchdog-timeout 9000 --context-length 8192 \
#        --prefill-round-robin-balance --enable-dp-lm-head --tokenizer-worker-num 4 --enable-dp-attention
#        NODE_RANK=$i
#        break
#    fi
#done

#--disable-cuda-graph
#        --ep-dispatch-algorithm static --init-expert-location /home/chenxu/sglang_ascend_1009/sglang_ascend/hot_map/expert_distribution_recorder_1762744462.8401136.pt \
#--ep-dispatch-algorithm static --init-expert-location /home/chenxu/sglang_ascend_1009/sglang_ascend/hot_map/expert_distribution_recorder_1762480358.327818.pt --ep-num-redundant-experts 32 \
#         --ep-dispatch-algorithm static --init-expert-location /home/lws/hot_map/expert_distribution_recorder_1760620441.849726.pt \
# ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf  --debug --summarizer default_perf --mode perf --num-prompts 1536
# vim ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
# --ep-dispatch-algorithm static --init-expert-location /home/lws/hot_map/expert_distribution_recorder_1760671625.6899254.pt \
#         --ep-dispatch-algorithm static --init-expert-location /home/chenxu/sglang_ascend_1009/sglang_ascend/hot_map/expert_distribution_recorder_1762480358.327818.pt \
#python3 -m sglang.bench_serving \
#    --dataset-path /home/chenxu/benchmark/ais_bench/datasets/gsm8k/GSM8K-in8192-bs1000.jsonl \
#    --dataset-name gsm8k --backend sglang \
#    --host 127.0.0.1 --port 6688 \
#    --max-concurrency 20 --random-output-len 200 --random-input-len 12000 --num-prompts 20
exit 1
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://141.61.105.141:8000 8995 \
    --decode http://141.61.105.143:8001 \
    --host 127.0.0.1 \
    --port 6688 \
    --mini-lb
