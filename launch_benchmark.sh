#!/bin/bash
set -xe

# MiDaS
function main {
    # prepare workload
    workload_dir="${PWD}"
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    if [ ! -e model-small-70d6b9c8.pt ];then
        tar xvf /home2/pytorch-broad-models/midasnet/dataset/input.tar.gz -C ./input/
    fi
    # requirements
    if [ "${CKPT_DIR}" == "" ];then
        set +x
        echo "[ERROR] Please set CKPT_DIR before launch"
        echo "  export CKPT_DIR=/path/to/checkpoint"
        exit 1
        set -x
    fi
    if [ "${device}" == "cuda" ];then
        pip install opencv-python==4.8.0.74
    fi

    if [ ! -d rwightman_gen-efficientnet-pytorch_master ];then
        git clone https://github.com/rwightman/gen-efficientnet-pytorch.git rwightman_gen-efficientnet-pytorch_master
        cd rwightman_gen-efficientnet-pytorch_master
        git checkout a4ac4dd7d72069a2aa4564df53ff4eb9b9f64400
        patch -f -p1 < ${workload_dir}/container_abcs.patch || true
        cd ..
        cache_dir="${HOME}/.cache/torch/hub"
        mkdir -p ${cache_dir} && rm -rf ${cache_dir}/rwightman_gen-efficientnet-pytorch_master
        cp -r rwightman_gen-efficientnet-pytorch_master ${cache_dir}
    fi

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # pre run
        python run.py --model_weights ${CKPT_DIR} --model_type small --arch midasnet \
            --warmup_iterations 1 --num_iterations 2 \
            --precision ${precision} --channels_last ${channels_last} || true
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean
            # generate launch script for multiple instance
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi

        printf " ${OOB_EXEC_HEADER} \
            python run.py --model_weights ${CKPT_DIR} --model_type small --arch midasnet \
                --warmup_iterations ${num_warmup} --num_iterations ${num_iter} \
                --precision ${precision} \
                --channels_last ${channels_last} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# run
function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#cpu_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            run.py --model_weights ${CKPT_DIR} --model_type small --arch midasnet \
                --warmup_iterations ${num_warmup} --num_iterations ${num_iter} \
                --precision ${precision} \
                --channels_last ${channels_last} \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git

# Start
main "$@"
