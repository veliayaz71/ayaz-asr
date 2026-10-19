#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalise a sherpa-onnx int8 package (encoder/decoder/joiner.int8.onnx + tokens.txt).

For each model file: the 'comment' metadata value is replaced by a neutral text (--comment), the file must be a single
file (no external data), metadata / graph / node names are scanned for forbidden patterns (--forbid; e.g. your
home directory or user name) before and after, and the file must open in onnxruntime (CPU).
Against the official sherpa-onnx package of the base model (--reference): the encoder's required metadata keys
(everything except 'comment') must be identical (18 keys expected) and tokens.txt must be byte-identical (the
vocabulary must not change).

Needs only onnx and onnxruntime (see export/requirements-export.txt).

TR: sherpa-onnx int8 paketinin son denetimi: 'comment' üst verisi nötr metne çevrilir, yasak desen taraması yapılır,
resmî paketle zorunlu üst veri anahtarları ve tokens.txt karşılaştırılır.
"""
import argparse
import filecmp
import os
import re
import shutil
import sys

MODELS = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx")
DEFAULT_COMMENT = ("fine-tuned from nvidia/nemotron-3.5-asr-streaming-0.6b; exported with sherpa-onnx "
                   "export_onnx.py (560 ms chunks); int8 dynamic quantization")


class Stop(Exception):
    pass


def nodes(g, onnx):
    for n in g.node:
        yield n
        for at in n.attribute:
            if at.type == onnx.AttributeProto.GRAPH:
                yield from nodes(at.g, onnx)
            elif at.type == onnx.AttributeProto.GRAPHS:
                for sub in at.graphs:
                    yield from nodes(sub, onnx)


def scan(m, patterns, onnx):
    found = {}

    def add(field, text):
        n = sum(len(rx.findall(text)) for rx in patterns) if text else 0
        if n:
            found[field] = found.get(field, 0) + n

    add("model doc_string", m.doc_string)
    add("producer_name", m.producer_name)
    add("producer_version", m.producer_version)
    add("domain", m.domain)
    add("graph name", m.graph.name)
    add("graph doc_string", m.graph.doc_string)
    for p in m.metadata_props:
        add("metadata key", p.key)
        add("metadata: " + p.key, p.value)
    for n in nodes(m.graph, onnx):
        add("node name", n.name)
        add("node doc_string", n.doc_string)
    return found


def external(m, onnx):
    return any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)


def process(name, src_dir, dst_dir, comment, patterns):
    import onnx
    import onnxruntime as ort
    src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
    m = onnx.load(src, load_external_data=False)
    if external(m, onnx):
        raise Stop(f"{name} carries external data (a single file is expected)")
    before = scan(m, patterns, onnx)
    n_meta = len(m.metadata_props)
    had_comment = any(p.key == "comment" for p in m.metadata_props)
    changed = False
    for p in m.metadata_props:
        if p.key == "comment" and p.value != comment:
            p.value = comment
            changed = True
    if changed:
        tmp = dst + ".writing"
        onnx.save_model(m, tmp, save_as_external_data=False)
        os.replace(tmp, dst)
    else:
        shutil.copyfile(src, dst)
    del m
    m2 = onnx.load(dst, load_external_data=False)
    if external(m2, onnx):
        raise Stop(f"{name} carries external data after saving")
    after = scan(m2, patterns, onnx)
    meta = dict((p.key, p.value) for p in m2.metadata_props)
    del m2
    if had_comment and meta.get("comment") != comment:
        raise Stop(f"{name}: comment could not be set")
    state = "replaced" if changed else ("already neutral" if had_comment else "none")
    print(f"  {name}: metadata {n_meta} keys · comment {state} · forbidden matches before {sum(before.values())} -> "
          f"after {sum(after.values())}", flush=True)
    if after:
        raise Stop(f"{name}: {sum(after.values())} forbidden matches after the change ({after})")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(dst, sess_options=so, providers=["CPUExecutionProvider"])
    n_in, n_out = len(sess.get_inputs()), len(sess.get_outputs())
    del sess
    print(f"    OK {'rewritten with onnx.save' if changed else 'copied byte for byte'} · {os.path.getsize(dst)} bytes · "
          f"no external data · opens in ORT CPU (inputs {n_in} · outputs {n_out})", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory with *.int8.onnx and tokens.txt / kaynak dizin")
    ap.add_argument("--dst", required=True, help="output package directory (created) / çıktı dizini")
    ap.add_argument("--reference", required=True,
                    help="official sherpa-onnx package of the base model (encoder.int8.onnx, tokens.txt) / resmî paket")
    ap.add_argument("--comment", default=DEFAULT_COMMENT, help="neutral 'comment' metadata (default: %(default)s)")
    ap.add_argument("--forbid", action="append", default=[r"/(?:Users|home)/"],
                    help="regex that must not appear in metadata or names, case-insensitive; repeatable "
                         "(default: macOS/Linux home-directory paths; add your user name)")
    a = ap.parse_args()
    import onnx
    patterns = [re.compile(p, re.I) for p in a.forbid]
    os.makedirs(a.dst, exist_ok=True)
    try:
        results = {name: process(name, a.src, a.dst, a.comment, patterns) for name in MODELS}
        r = onnx.load(os.path.join(a.reference, "encoder.int8.onnx"), load_external_data=False)
        rm = dict((p.key, p.value) for p in r.metadata_props)
        del r
        bm = results["encoder.int8.onnx"]
        ra = {k: v for k, v in rm.items() if k != "comment"}
        ba = {k: v for k, v in bm.items() if k != "comment"}
        missing, extra = sorted(set(ra) - set(ba)), sorted(set(ba) - set(ra))
        differ = sorted(k for k in set(ra) & set(ba) if ra[k] != ba[k])
        if missing or extra or differ:
            raise Stop(f"required keys differ from the official encoder · missing: {','.join(missing) or '-'} · "
                       f"extra: {','.join(extra) or '-'} · different: {','.join(differ) or '-'}")
        print(f"  OK required metadata identical to the official encoder: {len(ra)} keys (only 'comment' may differ)", flush=True)
        if len(ra) != 18:
            print(f"  ! 18 required keys were expected, found {len(ra)}", flush=True)
        shutil.copyfile(os.path.join(a.src, "tokens.txt"), os.path.join(a.dst, "tokens.txt"))
        if not filecmp.cmp(os.path.join(a.dst, "tokens.txt"), os.path.join(a.reference, "tokens.txt"), shallow=False):
            raise Stop("tokens.txt is not byte-identical to the official package")
        print(f"  OK tokens.txt identical to the official package · {os.path.getsize(os.path.join(a.dst, 'tokens.txt'))} bytes", flush=True)
        extra_files = sorted(set(os.listdir(a.dst)) - set(MODELS + ("tokens.txt",)))
        if extra_files:
            raise Stop(f"unexpected file in the package directory: {', '.join(extra_files)}")
        print(f"  OK {os.path.basename(os.path.abspath(a.dst))}: 4 files", flush=True)
    except Stop as e:
        print(f"  STOP: {e}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
