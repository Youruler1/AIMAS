# -*- coding: utf-8 -*-
import shutil
import sys
import gradio as gr
from pydub import AudioSegment
import os
import tempfile
import shutil
import subprocess


now_dir = os.getcwd()
sys.path.append(now_dir)
import traceback, pdb
import warnings

import numpy as np
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
import threading
from random import shuffle
from subprocess import Popen
from time import sleep

import faiss
import ffmpeg
import gradio as gr
import soundfile as sf
from config import Config
from fairseq import checkpoint_utils
from i18n import I18nAuto
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from infer_uvr5 import _audio_pre_, _audio_pre_new
from my_utils import load_audio
from train.process_ckpt import change_info, extract_small_model, merge, show_info
from vc_infer_pipeline import VC
from sklearn.cluster import MiniBatchKMeans

logging.getLogger("numba").setLevel(logging.WARNING)


tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(
    "%s/runtime/Lib/site-packages/lib.infer_pack" % (now_dir), ignore_errors=True
)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
i18n = I18nAuto()
i18n.print()
# Check if there is an NVIDIA GPU available for training and accelerating inference
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # At least one usable NVIDIA GPU
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n(
        "Unfortunately, you don’t have a usable GPU here to support training."
    )
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


hubert_model = None

# TOOLS KA FUNCTIONS


def safe_load_audio(input_path):
    input_path = input_path.name
    converted = input_path + "_pcm.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            converted,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return AudioSegment.from_file(converted)


def split_and_zip(uploaded_file, target_rate):
    target_rate = int(target_rate)
    audio = safe_load_audio(uploaded_file)
    audio = audio.set_frame_rate(target_rate)
    chunk_duration_ms = 10 * 1000
    temp_dir = tempfile.mkdtemp()
    chunks_dir = os.path.join(temp_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    chunk_files = []

    for i, start in enumerate(range(0, len(audio), chunk_duration_ms)):
        end = min(start + chunk_duration_ms, len(audio))
        chunk = audio[start:end]
        chunk_filename = os.path.join(chunks_dir, f"chunk_{i+1}.wav")
        chunk.export(chunk_filename, format="wav")
        chunk_files.append(chunk_filename)

    zip_path = os.path.join(temp_dir, "audio_chunks.zip")
    shutil.make_archive(
        base_name=zip_path.replace(".zip", ""), format="zip", root_dir=chunks_dir
    )
    return zip_path, chunk_files


def convert_format(uploaded_file, target_format):
    audio = safe_load_audio(uploaded_file)
    temp_dir = tempfile.mkdtemp()
    converted_path = os.path.join(temp_dir, f"converted.{target_format}")
    audio.export(converted_path, format=target_format)
    return converted_path


def trim_audio(uploaded_file, start_sec, end_sec):
    audio = safe_load_audio(uploaded_file)
    trimmed = audio[int(start_sec * 1000) : int(end_sec * 1000)]
    temp_dir = tempfile.mkdtemp()
    trimmed_path = os.path.join(temp_dir, "trimmed_audio.wav")
    trimmed.export(trimmed_path, format="wav")
    return trimmed_path


def combine_audio(file_list):
    combined = None
    for file_path in file_list:
        segment = safe_load_audio(file_path)
        combined = segment if combined is None else combined + segment
    temp_dir = tempfile.mkdtemp()
    combined_path = os.path.join(temp_dir, "combined_audio.wav")
    combined.export(combined_path, format="wav")
    return combined_path


def _split_and_return(f, rate):
    zip_path, chunks = split_and_zip(f, rate)
    preview = chunks[0] if chunks else None
    return zip_path, preview


def _convert_and_return(f, fmt):
    path = convert_format(f, fmt)
    return path, path


def _trim_and_return(f, s, e):
    path = trim_audio(f, s, e)
    return path, path


def _combine_and_return(files):
    path = combine_audio(files)
    return path, path


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # Prevent beginners from making mistakes, automatically replace it for them
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1,
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # Prevent newbies from copying paths with leading/trailing spaces, quotes (") and newline characters.
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                # file_big_npy,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    if format1 in ["wav", "flac"]:
                        sf.write(
                            "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                            audio_opt,
                            tgt_sr,
                        )
                    else:
                        path = "%s/%s.wav" % (opt_root, os.path.basename(path))
                        sf.write(
                            path,
                            audio_opt,
                            tgt_sr,
                        )
                        if os.path.exists(path):
                            os.system(
                                "ffmpeg -i %s -vn %s -q:a 2 -y"
                                % (path, path[:-4] + ".%s" % format1)
                            )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            from MDXNet import MDXNetDereverb

            pre_fun = MDXNetDereverb(15)
        else:
            func = _audio_pre_ if "DeEcho" not in model_name else _audio_pre_new
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


# Each tab can only have one timbre (voice).
def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if (
            hubert_model is not None
        ):  # Considering polling, we need to add a check to see whether sid has switched from having a model to not having one.
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###Otherwise, if we don’t do this cleanup thoroughly, it won’t be clean.
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll == None indicates that the process has not ended.
        # As long as there is even one process that hasn’t finished, don’t stop.
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
        % (trainset_dir, sr, n_p, now_dir, exp_dir)
        + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)
    # , stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
    # In Gradio, Popen.read will wait until the process finishes
    # before dumping all output at once.
    # In a normal environment, you can read line by line in real-time.
    # Therefore, we need to create an extra text stream, read stdout
    # periodically/asynchronously, to achieve real-time output.
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ### Stupid Gradio — with Popen, read() will only give all the output at once
        ### after the process fully finishes. Without Gradio, it normally reads line
        ### by line in real time. So the only workaround is to create an extra text
        ### stream and read it periodically.

        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        print(log)
        yield log
    #### Start multiple processes separately for different parts

    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            config.python_cmd
            + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
            % (
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ### Damn Gradio, popen read insists on finishing the whole run before dumping output all at once; without Gradio it reads line by line normally. So we have to create an extra text stream to read periodically.
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access(
        "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        print(
            "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2),
            "not exist, will not use pretrained model",
        )
    if not if_pretrained_discriminator_exist:
        print(
            "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2),
            "not exist, will not use pretrained model",
        )
    return (
        (
            "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access(
        "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        print(
            "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2),
            "not exist, will not use pretrained model",
        )
    if not if_pretrained_discriminator_exist:
        print(
            "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2),
            "not exist, will not use pretrained model",
        )
    return (
        (
            "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    if_pretrained_generator_exist = os.access(
        "pretrained%s/f0G%s.pth" % (path_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "pretrained%s/f0D%s.pth" % (path_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        print(
            "pretrained%s/f0G%s.pth" % (path_str, sr2),
            "not exist, will not use pretrained model",
        )
    if not if_pretrained_discriminator_exist:
        print(
            "pretrained%s/f0D%s.pth" % (path_str, sr2),
            "not exist, will not use pretrained model",
        )
    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            (
                "pretrained%s/f0G%s.pth" % (path_str, sr2)
                if if_pretrained_generator_exist
                else ""
            ),
            (
                "pretrained%s/f0D%s.pth" % (path_str, sr2)
                if if_pretrained_discriminator_exist
                else ""
            ),
        )
    return (
        {"visible": False, "__type__": "update"},
        (
            ("pretrained%s/G%s.pth" % (path_str, sr2))
            if if_pretrained_generator_exist
            else ""
        ),
        (
            ("pretrained%s/D%s.pth" % (path_str, sr2))
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # Generate file list

    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # Generate config # or No need to generate config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "\b",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "\b",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "Training complete. You can check the training logs in the console or in the train.log file under the experiment folder."


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "Please perform feature extraction first!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "Please perform feature extraction first! "
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        # if(1):
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Index built successfully: added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir
    feature_dir = (
        "%s/3_feature256" % model_log_dir
        if version19 == "v1"
        else "%s/3_feature768" % model_log_dir
    )

    os.makedirs(model_log_dir, exist_ok=True)
    ######### step1: process data
    open(preprocess_log_path, "w").close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s "
        % (trainset_dir4, sr_dict[sr2], np7, model_log_dir)
        + str(config.noparallel)
    )
    yield get_info_str(i18n("step1: Processing data"))
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open(preprocess_log_path, "r") as f:
        print(f.read())

    ######### step2a: Pitch Extraction
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        yield get_info_str("step2a: Extracting pitch...")
        cmd = config.python_cmd + " extract_f0_print.py %s %s %s" % (
            model_log_dir,
            np7,
            f0method8,
        )
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        with open(extract_f0_feature_log_path, "r") as f:
            print(f.read())
    else:
        yield get_info_str(i18n("step2a: Skipping pitch extraction"))

    ####### step2b: Feature Extraction
    yield get_info_str(i18n("step2b: Extracting features..."))

    gpus = gpus16.split("-")
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s %s" % (
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
            version19,
        )
        yield get_info_str(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    for p in ps:
        p.wait()
    with open(extract_f0_feature_log_path, "r") as f:
        print(f.read())
    ####### step3a: Train Model
    yield get_info_str(i18n("step3a: Training model..."))
    # Generate filelist

    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    yield get_info_str("write filelist done")
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    yield get_info_str(
        i18n(
            "Training finished, you can check the training logs in the console or in train.log under the experiment folder"
        )
    )
    ####### step3b: Train Index

    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        # if(1):
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        yield get_info_str(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            print(info)
            yield get_info_str(info)

    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    yield get_info_str("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    yield get_info_str("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    yield get_info_str(
        "Index successfully built, added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    yield get_info_str(i18n("全流程结束！"))


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    test_phone = torch.rand(1, 200, vec_channels)  # hidden unit
    test_phone_lengths = torch.tensor(
        [200]
    ).long()  # length of hidden unit (seems not very useful)
    test_pitch = torch.randint(
        size=(1, 200), low=5, high=255
    )  # fundamental frequency (in Hz)
    test_pitchf = torch.rand(1, 200)  # nsf fundamental frequency
    test_ds = torch.LongTensor([0])  # speaker ID
    test_rnd = torch.rand(1, 192, 200)  # noise (adds randomness factor)

    device = "cpu"  # Device used during export (does not affect model usage)

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False, version=cpt.get("version", "v1")
    )  # Export with fp32 (C++ requires manual memory reordering to support fp16, so fp16 is not used for now)
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker)  multi-speaker mixed track export
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    return "Finished"


with gr.Blocks() as app:
    # Welcome page
    gr.Markdown("### Welcome to RVC")
    with gr.Row():
        rvc_button = gr.Button("RVC", variant="primary")
        tools_button = gr.Button("TOOLS", variant="secondary")

    # RVC Interface (hidden by default, shown when RVC button is clicked)
    with gr.Column(visible=False) as rvc_interface:
        # gr.Markdown( # THIS WAS USELESS TEXT
        #     value=i18n(
        #         "This software is open-sourced under the MIT license. The author has no control over the software. Users of the software and those distributing voices generated with it are solely responsible. <br>If you do not agree with this clause, you cannot use or reference any code and files within this package. See the root directory <b>LICENSE</b> for details."
        #     )
        # )
        with gr.Tabs():
            with gr.TabItem(
                i18n("Model Inference")
            ):  # THIS IS THE MODEL INFERENCE UI PART
                with gr.Row():
                    sid0 = gr.Dropdown(
                        label=i18n("Inference Voice"), choices=sorted(names)
                    )
                    refresh_button = gr.Button(
                        i18n("Refresh voice list and index path"), variant="primary"
                    )
                    clean_button = gr.Button(
                        i18n("Unload voice to save VRAM"), variant="primary"
                    )
                    spk_item = gr.Slider(
                        minimum=0,
                        maximum=2333,
                        step=1,
                        label=i18n("Select speaker ID"),
                        value=0,
                        visible=False,
                        interactive=True,
                    )
                    clean_button.click(fn=clean, inputs=[], outputs=[sid0])
                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "Male to female recommended +12 key, female to male recommended -12 key. If vocal range explosion causes timbre distortion, you can adjust to a suitable range yourself."
                        )
                    )
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(
                                label=i18n(
                                    "Pitch shift (integer, semitone count, +12 for one octave up, -12 for one octave down)"
                                ),
                                value=0,
                            )
                            input_audio0 = gr.Textbox(
                                label=i18n(
                                    "Path of input audio file to process (default is a correct format example)"
                                ),
                                value="E:\\codes\\py39\\test-20230416b\\todo-songs\\winter_flower_clip1.wav",
                            )
                            f0method0 = gr.Radio(
                                label=i18n(
                                    "Choose pitch extraction algorithm. For singing input, pm speeds up; harvest handles low pitch well but is very slow; crepe works well but uses GPU"
                                ),
                                choices=["pm", "harvest", "crepe", "rmvpe"],
                                value="pm",
                                interactive=True,
                            )
                            filter_radius0 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=i18n(
                                    ">=3 applies median filtering to harvest pitch results. The number is the filter radius. Using it can weaken hoarseness"
                                ),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                        with gr.Column():
                            file_index1 = gr.Textbox(
                                label=i18n(
                                    "Feature index library file path. Leave empty to use dropdown selection result"
                                ),
                                value="",
                                interactive=True,
                            )
                            file_index2 = gr.Dropdown(
                                label=i18n(
                                    "Auto-detected index path, dropdown selection"
                                ),
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            refresh_button.click(
                                fn=change_choices,
                                inputs=[],
                                outputs=[sid0, file_index2],
                            )
                            index_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("Feature search proportion"),
                                value=0.75,
                                interactive=True,
                            )
                        with gr.Column():
                            resample_sr0 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n(
                                    "Resample after processing to final sample rate. 0 means no resampling"
                                ),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate0 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n(
                                    "Mix ratio of replacing source volume envelope with output envelope. Closer to 1 uses output envelope more"
                                ),
                                value=0.25,
                                interactive=True,
                            )
                            protect0 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "Protect unvoiced consonants and breaths to prevent electronic tearing artifacts. At max 0.5 it is off; lowering increases protection strength but may reduce index effectiveness"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                        f0_file = gr.File(
                            label=i18n(
                                "F0 curve file, optional. One pitch per line, replaces default F0 and pitch shift"
                            )
                        )
                        but0 = gr.Button(i18n("Convert"), variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label=i18n("Output info"))
                            vc_output2 = gr.Audio(
                                label=i18n(
                                    "Output audio (Click bottom right three dots to download)"
                                )
                            )
                        but0.click(
                            vc_single,
                            [
                                spk_item,
                                input_audio0,
                                vc_transform0,
                                f0_file,
                                f0method0,
                                file_index1,
                                file_index2,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ],
                            [vc_output1, vc_output2],
                        )
                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "Batch conversion. Input the folder of audio files to convert, or upload multiple audio files. The converted audio will be output to the specified folder (default opt)."
                        )
                    )
                    with gr.Row():
                        with gr.Column():
                            vc_transform1 = gr.Number(
                                label=i18n(
                                    "Pitch shift (integer, semitone count, +12 for one octave up, -12 for one octave down)"
                                ),
                                value=0,
                            )
                            opt_input = gr.Textbox(
                                label=i18n("Specify output folder"), value="opt"
                            )
                            f0method1 = gr.Radio(
                                label=i18n(
                                    "Choose pitch extraction algorithm. For singing input, pm speeds up; harvest handles low pitch well but is very slow; crepe works well but uses GPU"
                                ),
                                choices=["pm", "harvest", "crepe", "rmvpe"],
                                value="pm",
                                interactive=True,
                            )
                            filter_radius1 = gr.Slider(
                                minimum=0,
                                maximum=7,
                                label=i18n(
                                    ">=3 applies median filtering to harvest pitch results. The number is the filter radius. Using it can weaken hoarseness"
                                ),
                                value=3,
                                step=1,
                                interactive=True,
                            )
                        with gr.Column():
                            file_index3 = gr.Textbox(
                                label=i18n(
                                    "Feature index library file path. Leave empty to use dropdown selection result"
                                ),
                                value="",
                                interactive=True,
                            )
                            file_index4 = gr.Dropdown(
                                label=i18n(
                                    "Auto-detected index path, dropdown selection"
                                ),
                                choices=sorted(index_paths),
                                interactive=True,
                            )
                            refresh_button.click(
                                fn=lambda: change_choices()[1],
                                inputs=[],
                                outputs=file_index4,
                            )
                            index_rate2 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n("Feature search proportion"),
                                value=1,
                                interactive=True,
                            )
                        with gr.Column():
                            resample_sr1 = gr.Slider(
                                minimum=0,
                                maximum=48000,
                                label=i18n(
                                    "Resample after processing to final sample rate. 0 means no resampling"
                                ),
                                value=0,
                                step=1,
                                interactive=True,
                            )
                            rms_mix_rate1 = gr.Slider(
                                minimum=0,
                                maximum=1,
                                label=i18n(
                                    "Mix ratio of replacing source volume envelope with output envelope. Closer to 1 uses output envelope more"
                                ),
                                value=1,
                                interactive=True,
                            )
                            protect1 = gr.Slider(
                                minimum=0,
                                maximum=0.5,
                                label=i18n(
                                    "Protect unvoiced consonants and breaths to prevent electronic tearing artifacts. At max 0.5 it is off; lowering increases protection strength but may reduce index effectiveness"
                                ),
                                value=0.33,
                                step=0.01,
                                interactive=True,
                            )
                        with gr.Column():
                            dir_input = gr.Textbox(
                                label=i18n(
                                    "Path of folder containing audio files to process (copy from file manager address bar)"
                                ),
                                value="E:\\codes\\py39\\test-20230416b\\todo-songs",
                            )
                            inputs = gr.File(
                                file_count="multiple",
                                label=i18n(
                                    "Alternatively, batch input audio files. Either-or. Folder has priority"
                                ),
                            )
                        with gr.Row():
                            format1 = gr.Radio(
                                label=i18n("Export file format"),
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="flac",
                                interactive=True,
                            )
                            but1 = gr.Button(i18n("Convert"), variant="primary")
                            vc_output3 = gr.Textbox(label=i18n("Output info"))
                        but1.click(
                            vc_multi,
                            [
                                spk_item,
                                dir_input,
                                opt_input,
                                inputs,
                                vc_transform1,
                                f0method1,
                                file_index3,
                                file_index4,
                                index_rate2,
                                filter_radius1,
                                resample_sr1,
                                rms_mix_rate1,
                                protect1,
                                format1,
                            ],
                            [vc_output3],
                        )
                sid0.change(
                    fn=get_vc,
                    inputs=[sid0, protect0, protect1],
                    outputs=[spk_item, protect0, protect1],
                )

            with gr.TabItem(i18n("Training")):  # THIS IS THE TRAIN TAB UI PART
                gr.Markdown(
                    value=i18n(
                        "step1: Fill in experiment configuration. Training data is placed in logs. Each experiment has its own folder containing configuration, logs, and models obtained from training. You must manually input the experiment name path."
                    )
                )
                with gr.Row():
                    exp_dir1 = gr.Textbox(
                        label=i18n("Input experiment name"), value="mi-test"
                    )
                    sr2 = gr.Radio(
                        label=i18n("Target sample rate"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_3 = gr.Radio(
                        label=i18n(
                            "Does the model use pitch guidance (required for singing, optional for speech)"
                        ),
                        choices=[True, False],
                        value=True,
                        interactive=True,
                    )
                    version19 = gr.Radio(
                        label=i18n("Version"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                        visible=True,
                    )
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label=i18n(
                            "Number of CPU processes used for pitch extraction and data processing"
                        ),
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                with gr.Group():  # Temporarily for single speaker. Later supports up to 4 speakers. # Data processing
                    gr.Markdown(
                        value=i18n(
                            "step2a: Automatically traverse all decodable audio files in training folder and perform slicing normalization. Generates 2 wav folders in experiment directory. Temporarily only supports single speaker training."
                        )
                    )
                    with gr.Row():
                        trainset_dir4 = gr.Textbox(
                            label=i18n("Input training folder path"),
                            value="E:\\SpeechAudio+Annotations\\YonezuKenshi\\src",
                        )
                        spk_id5 = gr.Slider(
                            minimum=0,
                            maximum=4,
                            step=1,
                            label=i18n("Specify speaker ID"),
                            value=0,
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("Process data"), variant="primary")
                        info1 = gr.Textbox(label=i18n("Output info"), value="")
                        but1.click(
                            preprocess_dataset,
                            [trainset_dir4, exp_dir1, sr2, np7],
                            [info1],
                        )

                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "step2b: Use CPU to extract pitch (if model uses pitch), GPU to extract features (choose GPU IDs)"
                        )
                    )
                    with gr.Row():
                        with gr.Column():
                            gpus6 = gr.Textbox(
                                label=i18n(
                                    "Input GPU IDs separated by dash, e.g. 0-1-2 uses GPUs 0,1,2"
                                ),
                                value=gpus,
                                interactive=True,
                            )
                            gpu_info9 = gr.Textbox(
                                label=i18n("GPU info"), value=gpu_info
                            )
                        with gr.Column():
                            f0method8 = gr.Radio(
                                label=i18n(
                                    "Choose pitch extraction algorithm: For singing input, pm speeds up; for high-quality speech but weak CPU, dio speeds up; harvest better quality but slower"
                                ),
                                choices=["pm", "harvest", "dio"],
                                value="harvest",
                                interactive=True,
                            )
                        but2 = gr.Button(i18n("Extract features"), variant="primary")
                        info2 = gr.Textbox(
                            label=i18n("Output info"), value="", max_lines=8
                        )
                        but2.click(
                            extract_f0_feature,
                            [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19],
                            [info2],
                        )

                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "step3: Fill in training settings and start training model and index"
                        )
                    )
                    with gr.Row():
                        save_epoch10 = gr.Slider(
                            minimum=0,
                            maximum=50,
                            step=1,
                            label=i18n("Save frequency (save_every_epoch)"),
                            value=5,
                            interactive=True,
                        )
                        total_epoch11 = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            step=1,
                            label=i18n("Total training epochs (total_epoch)"),
                            value=20,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("Batch size per GPU"),
                            value=default_batch_size,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Radio(
                            label=i18n(
                                "Save only latest checkpoint to save disk space"
                            ),
                            choices=[i18n("Yes"), i18n("No")],
                            value=i18n("No"),
                            interactive=True,
                        )
                        if_cache_gpu17 = gr.Radio(
                            label=i18n(
                                "Cache all training set to GPU memory. For small datasets (<10min), caching can speed up. Large datasets may cause OOM and won’t add much speed"
                            ),
                            choices=[i18n("Yes"), i18n("No")],
                            value=i18n("No"),
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Radio(
                            label=i18n(
                                "Save final small model to weights folder at every save point"
                            ),
                            choices=[i18n("Yes"), i18n("No")],
                            value=i18n("No"),
                            interactive=True,
                        )
                    with gr.Row():
                        pretrained_G14 = gr.Textbox(
                            label=i18n("Path to pre-trained base G"),
                            value="pretrained/f0G40k.pth",
                            interactive=True,
                        )
                        pretrained_D15 = gr.Textbox(
                            label=i18n("Path to pre-trained base D"),
                            value="pretrained/f0D40k.pth",
                            interactive=True,
                        )
                        sr2.change(
                            change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )

                        version19.change(
                            change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, sr2],
                        )
                        if_f0_3.change(
                            change_f0,
                            [if_f0_3, sr2, version19],
                            [f0method8, pretrained_G14, pretrained_D15],
                        )
                        gpus16 = gr.Textbox(
                            label=i18n(
                                "Enter GPU IDs separated by '-', e.g. 0-1-2 for using GPU 0, 1, and 2"
                            ),
                            value=gpus,
                            interactive=True,
                        )
                        but3 = gr.Button(i18n("Train Model"), variant="primary")
                        but4 = gr.Button(i18n("Train Feature Index"), variant="primary")
                        but5 = gr.Button(i18n("One-Click Training"), variant="primary")
                        info3 = gr.Textbox(
                            label=i18n("Output Information"), value="", max_lines=10
                        )
                        but3.click(
                            click_train,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            info3,
                        )
                        but4.click(train_index, [exp_dir1, version19], info3)
                        but5.click(
                            train1key,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                trainset_dir4,
                                spk_id5,
                                np7,
                                f0method8,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            info3,
                        )
            with gr.TabItem(
                i18n("CKPT Processing")
            ):  # THIS IS THE CKPT PROCESSING UI PART
                with gr.Group():
                    gr.Markdown(
                        value=i18n("Model fusion, can be used to test timbre fusion")
                    )
                    with gr.Row():
                        ckpt_a = gr.Textbox(
                            label=i18n("Model A Path"), value="", interactive=True
                        )
                        ckpt_b = gr.Textbox(
                            label=i18n("Model B Path"), value="", interactive=True
                        )
                        alpha_a = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Model A Weight"),
                            value=0.5,
                            interactive=True,
                        )
                    with gr.Row():
                        sr_ = gr.Radio(
                            label=i18n("Target Sample Rate"),
                            choices=["40k", "48k"],
                            value="40k",
                            interactive=True,
                        )
                        if_f0_ = gr.Radio(
                            label=i18n("Does the model include pitch guidance"),
                            choices=[i18n("Yes"), i18n("No")],
                            value=i18n("Yes"),
                            interactive=True,
                        )
                        info__ = gr.Textbox(
                            label=i18n("Model Info to Insert"),
                            value="",
                            max_lines=8,
                            interactive=True,
                        )
                        name_to_save0 = gr.Textbox(
                            label=i18n("Name of Saved Model (without suffix)"),
                            value="",
                            max_lines=1,
                            interactive=True,
                        )
                        version_2 = gr.Radio(
                            label=i18n("Model Version Type"),
                            choices=["v1", "v2"],
                            value="v1",
                            interactive=True,
                        )
                    with gr.Row():
                        but6 = gr.Button(i18n("Merge"), variant="primary")
                        info4 = gr.Textbox(
                            label=i18n("Output Info"), value="", max_lines=8
                        )
                    but6.click(
                        merge,
                        [
                            ckpt_a,
                            ckpt_b,
                            alpha_a,
                            sr_,
                            if_f0_,
                            info__,
                            name_to_save0,
                            version_2,
                        ],
                        info4,
                    )  # def merge(path1, path2, alpha1, sr, f0, info):

                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "Edit Model Info (only supports small model files extracted under the weights folder)"
                        )
                    )
                    with gr.Row():
                        ckpt_path0 = gr.Textbox(
                            label=i18n("Model Path"), value="", interactive=True
                        )
                        info_ = gr.Textbox(
                            label=i18n("Model Info to Modify"),
                            value="",
                            max_lines=8,
                            interactive=True,
                        )
                        name_to_save1 = gr.Textbox(
                            label=i18n(
                                "Save Filename (default empty = same as source file)"
                            ),
                            value="",
                            max_lines=8,
                            interactive=True,
                        )
                    with gr.Row():
                        but7 = gr.Button(i18n("Modify"), variant="primary")
                        info5 = gr.Textbox(
                            label=i18n("Output Info"), value="", max_lines=8
                        )
                    but7.click(change_info, [ckpt_path0, info_, name_to_save1], info5)

                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "View Model Info (only supports small model files extracted under the weights folder)"
                        )
                    )
                    with gr.Row():
                        ckpt_path1 = gr.Textbox(
                            label=i18n("Model Path"), value="", interactive=True
                        )
                        but8 = gr.Button(i18n("View"), variant="primary")
                        info6 = gr.Textbox(
                            label=i18n("Output Info"), value="", max_lines=8
                        )
                    but8.click(show_info, [ckpt_path1], info6)

                with gr.Group():
                    gr.Markdown(
                        value=i18n(
                            "Extract Model (input large model path from logs folder). "
                            "Useful if training was stopped halfway and the model was not automatically extracted into a small file, "
                            "or if you want to test an intermediate model."
                        )
                    )
                    with gr.Row():
                        ckpt_path2 = gr.Textbox(
                            label=i18n("Model Path"),
                            value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                            interactive=True,
                        )
                        save_name = gr.Textbox(
                            label=i18n("Save Name"), value="", interactive=True
                        )
                        sr__ = gr.Radio(
                            label=i18n("Target Sample Rate"),
                            choices=["32k", "40k", "48k"],
                            value="40k",
                            interactive=True,
                        )
                        if_f0__ = gr.Radio(
                            label=i18n(
                                "Does the model include pitch guidance (1 = Yes, 0 = No)"
                            ),
                            choices=["1", "0"],
                            value="1",
                            interactive=True,
                        )
                        version_1 = gr.Radio(
                            label=i18n("Model Version Type"),
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                        )
                        info___ = gr.Textbox(
                            label=i18n("Model Info to Insert"),
                            value="",
                            max_lines=8,
                            interactive=True,
                        )
                        but9 = gr.Button(i18n("Extract"), variant="primary")
                        info7 = gr.Textbox(
                            label=i18n("Output Info"), value="", max_lines=8
                        )
                        ckpt_path2.change(
                            change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                        )
                    but9.click(
                        extract_small_model,
                        [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                        info7,
                    )

            # with gr.TabItem(i18n("Onnx导出")):
            #     with gr.Row():
            #         ckpt_dir = gr.Textbox(label=i18n("RVC模型路径"), value="", interactive=True)
            #     with gr.Row():
            #         onnx_dir = gr.Textbox(
            #             label=i18n("Onnx输出路径"), value="", interactive=True
            #         )
            #     with gr.Row():
            #         infoOnnx = gr.Label(label="info")
            #     with gr.Row():
            #         butOnnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
            #     butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx)

            # tab_faq = i18n("常见问题解答")
            # with gr.TabItem(tab_faq):
            #     try:
            #         if tab_faq == "常见问题解答":
            #             with open("docs/faq.md", "r", encoding="utf8") as f:
            #                 info = f.read()
            #         else:
            #             with open("docs/faq_en.md", "r", encoding="utf8") as f:
            #                 info = f.read()
            #         gr.Markdown(value=info)
            #     except:
            #         gr.Markdown(traceback.format_exc())

            # with gr.TabItem(i18n("招募音高曲线前端编辑器")):
            #     gr.Markdown(value=i18n("加开发群联系我xxxxx"))
            # with gr.TabItem(i18n("点击查看交流、问题反馈群号")):
            #     gr.Markdown(value=i18n("xxxxx"))

    # TOOLS Interface (hidden by default, shown when TOOLS button is clicked)
    with gr.Column(visible=False) as tools_interface:
        gr.Markdown("## 🎧 Audio Tools")

        with gr.Tab("Split + Resample"):
            file1 = gr.File(label="Upload MP3 or WAV", file_types=["audio"])
            target_rate = gr.Number(value=40000, label="Target sampling rate (Hz)")
            btn_split = gr.Button("Process")
            zip_out = gr.File(label="Download chunks ZIP")
            audio_out1 = gr.Audio(label="Preview first chunk")

            btn_split.click(
                _split_and_return, [file1, target_rate], [zip_out, audio_out1]
            )

        with gr.Tab("Convert Format"):
            file2 = gr.File(label="Upload audio", file_types=["audio"])
            fmt = gr.Dropdown(
                choices=["wav", "mp3", "flac", "ogg", "aac"],
                value="wav",
                label="Target format",
            )
            btn_convert = gr.Button("Convert")
            conv_out = gr.File(label="Download converted file")
            audio_out2 = gr.Audio(label="Preview")

            btn_convert.click(_convert_and_return, [file2, fmt], [conv_out, audio_out2])

        with gr.Tab("Trim Audio"):
            file3 = gr.File(label="Upload audio", file_types=["audio"])
            start = gr.Number(value=0.0, label="Start time (sec)")
            end = gr.Number(value=10.0, label="End time (sec)")
            btn_trim = gr.Button("Trim")
            trim_out = gr.File(label="Download trimmed file")
            audio_out3 = gr.Audio(label="Preview trimmed")

            btn_trim.click(
                _trim_and_return, [file3, start, end], [trim_out, audio_out3]
            )

        with gr.Tab("Combine Audio"):
            file4 = gr.File(
                label="Upload multiple audio files",
                file_types=["audio"],
                file_count="multiple",
            )
            btn_combine = gr.Button("Combine")
            comb_out = gr.File(label="Download combined file")
            audio_out4 = gr.Audio(label="Preview combined")

            btn_combine.click(_combine_and_return, file4, [comb_out, audio_out4])

    # Toggle between RVC and TOOLS interfaces
    rvc_button.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        inputs=[],
        outputs=[rvc_interface, tools_interface],
    )
    tools_button.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[rvc_interface, tools_interface],
    )

    # Launch the app
    app.queue(concurrency_count=511, max_size=1022).launch(share=True, inbrowser=True)
