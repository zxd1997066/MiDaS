"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import time

from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, model_type="large", jit=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("using ipex model to do inference\n")
        device = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "large":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
    elif model_type == "small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3",
                exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.eval()

    if args.channels_last:
        model_oob = model
        model_oob = model_oob.to(memory_format=torch.channels_last)
        model = model_oob
        print('---- Use channels last format.')
    else:
        model.to(device)
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})   
    if args.ipex:
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
            print("running ipex bf16 evalation step\n")
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
            print("running ipex fp32 evalation step\n")

    if jit == True:
        print("running jit fusion path\n")
        rand_example = torch.rand(1, 3, net_h, net_w).to(device)
        model(rand_example)
        traced_script_module = torch.jit.trace(model, rand_example)
        model = traced_script_module
        if args.ipex:
            model = torch.jit.freeze(model)
    
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)  
            model = model.half()

        # warm-up
        for i in range(10):
            model(rand_example)

    if args.trace:
        from torch.fx import symbolic_trace
        traced_dir = str(os.path.dirname(os.path.realpath(__file__))) + '/traced_model/'
        if not os.path.exists(traced_dir):
            os.makedirs(traced_dir)
        traced_path = traced_dir + args.arch + "_fx_traced_model.pth"
        # fx
        try:
            fx_traced = symbolic_trace(model)
            torch.save(fx_traced, traced_path)
        except:
            print("WARN: {} don't support FX trace.".format(args.arch))
        # jit
        traced_path = traced_dir + args.arch + "_jit_traced_model.pth"
        try:
            q_model = torch.jit.script(model_.eval())
            q_model.save(traced_path)
        except:
            try:
                q_model = torch.jit.trace(model.eval(), images)
                q_model.save(traced_path)
            except:
                print("WARN: {} don't support JIT script/trace.".format(args.arch))

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    all_time = 0
    all_images = 0
    batch_time_list = []
    
    for ind, img_name in enumerate(img_names):
        if args.num_iterations != 0 and ind > args.num_iterations:
            break
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]
        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if args.channels_last:
                sample = sample.to(memory_format=torch.channels_last)
            start = time.time()
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                    prediction = model.forward(sample)
                #
                if ind == int(args.num_iterations/2):
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        os.makedirs(timeline_dir)
                    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                args.arch + str(ind) + '-' + str(os.getpid()) + '.json'
                    print(timeline_file)
                    prof.export_chrome_trace(timeline_file)
                    table_res = prof.key_averages().table(sort_by="cpu_time_total")
                    print(table_res)
                    # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
            else:
                prediction = model.forward(sample)
            #prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2], 
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu().float().numpy()
            )
            end = time.time()
            print("Iteration: {}, inference time: {} sec.".format(ind, end - start), flush=True)
            if ind >= args.warmup_iterations:
                batch_time_list.append((end - start) * 1000)
                all_time += end - start
                all_images += 1

            # output
            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            utils.write_depth(filename, prediction, bits=2)

    print("\n", "-"*20, "Summary", "-"*20)
    latency = all_time / all_images * 1000
    throughput = all_images / all_time
    print("inference latency:\t {:.3f} ms".format(latency))
    print("inference Throughput:\t {:.2f} samples/s".format(throughput))
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))

    print("finished")


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default='input',
                        help='folder with input images'
                        )

    parser.add_argument('-o', '--output_path', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights', 
        default='model-f6b98070.pt',
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type', 
        default='large',
        help='model type: large or small'
    )

    parser.add_argument('--jit', dest='jit', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    parser.add_argument('--profile', action='store_true',help='help')
    parser.add_argument('--trace', action='store_true',help='help')
    parser.add_argument('--arch', type=str, help='model name')
    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    parser.add_argument('--precision', type=str, default="float32",
                        help='precision, float32, bfloat16')
    parser.add_argument('-w', '--warmup_iterations', default=5, type=int, metavar='N',
                        help='number of warmup iterations to run')
    parser.add_argument('--num_iterations', default=0, type=int, metavar='N',
                        help='number of iterations to run')
    parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")

    args = parser.parse_args()

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    if args.precision == "bfloat16":
        print('---- Enable AMP bfloat16')
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            run(args.input_path, args.output_path, args.model_weights, args.model_type, args.jit)
    elif args.precision == "float16":
        print('---- Enable AMP float16')
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
            run(args.input_path, args.output_path, args.model_weights, args.model_type, args.jit)
    else:
        run(args.input_path, args.output_path, args.model_weights, args.model_type, args.jit)
