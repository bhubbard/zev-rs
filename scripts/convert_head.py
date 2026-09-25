"""head.pt (torch zip pickle) -> head.safetensors + head_meta.json, without torch.

usage: python convert_head.py head.pt out_dir
"""
import json
import pickle
import struct
import sys
import zipfile

import numpy as np

DTYPES = {"FloatStorage": (np.float32, "F32"), "BFloat16Storage": (None, "BF16"), "HalfStorage": (np.float16, "F16")}


class Storage:
    def __init__(self, kind, key):
        self.kind, self.key = kind, key


def rebuild(storage, offset, size, stride, *_):
    return ("tensor", storage, offset, tuple(size), tuple(stride))


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return rebuild
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        _, kind, key, _loc, _n = pid
        return Storage(kind, key)


def main(path, out):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    obj = Unpickler(z.open(f"{root}/data.pkl")).load()
    tensors, meta = {}, {}

    def materialize(t):
        _, st, off, size, stride = t
        raw = z.read(f"{root}/data/{st.key}")
        np_dtype, tag = DTYPES[st.kind]
        arr = np.frombuffer(raw, dtype=np_dtype)
        n = int(np.prod(size)) if size else 1
        assert stride == tuple(int(np.prod(size[i + 1:])) for i in range(len(size))), "non-contiguous"
        return arr[off:off + n].reshape(size), tag

    for k, v in obj.items():
        if k == "head":
            for name, t in v.items():
                tensors[name] = materialize(t)
        else:
            meta[k] = v
    header, blobs, pos = {}, [], 0
    for name, (arr, tag) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [pos, pos + len(b)]}
        blobs.append(b)
        pos += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(f"{out}/head.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    json.dump(meta, open(f"{out}/head_meta.json", "w"), indent=1, default=str)
    print({k: (a.shape, a.dtype) for k, (a, _) in tensors.items()})
    print({k: v for k, v in meta.items() if k in ("base", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights_dtype", "temperature")})


if __name__ == "__main__":
    main(*sys.argv[1:3])
