#!/usr/bin/env python
"""Find a headless OpenGL backend that RoboCasa/MuJoCo can actually render with.

WHY THIS EXISTS
    robosuite picks its EGL device like this
    (robosuite/renderers/context/egl_context.py::create_initialized_egl_device_display):

        selected_device = MUJOCO_EGL_DEVICE_ID or CUDA_VISIBLE_DEVICES
        device_idx      = int(selected_device)
        candidates      = eglQueryDevicesEXT()[device_idx : device_idx + 1]

    i.e. it uses CUDA_VISIBLE_DEVICES as an INDEX INTO THE EGL DEVICE LIST -- two
    unrelated enumerations. Note the index is 0 both when CUDA_VISIBLE_DEVICES=0 and when
    it is unset (robosuite is called with render_gpu_device_id=-1), so a one-GPU Slurm job
    almost always asks for EGL device 0 regardless of cluster.

    There are therefore TWO distinct ways to hit

        ImportError: Cannot initialize a EGL device display...

    (a) WRONG INDEX: the requested index exists but is not a usable GPU device -- e.g. a
        multi-GPU node where Slurm exports a non-zero CUDA_VISIBLE_DEVICES. Fix:
        MUJOCO_EGL_DEVICE_ID=<a working index>.
    (b) NO USABLE NVIDIA EGL DEVICE AT ALL: the NVIDIA vendor contributes nothing to the
        list, so index 0 is a Mesa software device with no PLATFORM_DEVICE support. This
        is a driver-visibility problem, not an index problem, and MUJOCO_EGL_DEVICE_ID
        will not fix it. Tell-tales: no /dev/nvidia* in the job, or libEGL_nvidia.so.0
        not resolving.

    Alliance nodes install both an NVIDIA and a Mesa EGL vendor ICD
    (/usr/share/glvnd/egl_vendor.d/{10_nvidia,50_mesa}.json). Those files are VENDOR
    LIBRARIES, not devices: each vendor contributes 0..N devices, and the "10_"/"50_"
    prefixes set glvnd's vendor load order, NOT EGL device indices. In particular the
    NVIDIA vendor enumerates zero devices when no GPU is visible, however early it sorts.

    That surfaces only when the first eval env is constructed -- after the model has
    loaded, wandb has opened a run and the dataset has been fetched.

USAGE
    On a GPU node of the cluster in question:
        python scripts/check_gl_backend.py

    It prints every EGL device index, says which ones genuinely initialize, and emits the
    exact exports to put in your batch script. `--emit-exports` prints only those lines so
    a script can `eval` them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

# Probing has to happen in a subprocess per index: robosuite caches the initialized
# display in a module-global (EGL_DISPLAY), so a failed attempt poisons later ones.
_PROBE = textwrap.dedent(
    """
    import os, sys
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "nvidia-only":
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    try:
        from robosuite.renderers.context.egl_context import EGLGLContext, EGL_DISPLAY
        import robosuite.renderers.context.egl_context as ec
        EGLGLContext(max_width=64, max_height=64, device_id=-1)
        from OpenGL import EGL
        vendor = EGL.eglQueryString(ec.EGL_DISPLAY, EGL.EGL_VENDOR)
        print(f"OK vendor={vendor.decode() if hasattr(vendor, 'decode') else vendor}")
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}".replace("\\n", " ")[:200])
        sys.exit(1)
    """
)

# Direct enumeration, deliberately WITHOUT robosuite. robosuite's binding_utils asserts
# MUJOCO_EGL_DEVICE_ID is a substring of CUDA_VISIBLE_DEVICES, so with CUDA_VISIBLE_DEVICES=0
# it can only ever try EGL index 0 -- which tells you nothing about what the other devices
# are. This maps the real device list so you can see whether index 0 is the GPU or a Mesa
# software device that merely sorts first.
_ENUM_PROBE = textwrap.dedent(
    """
    import os, sys
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    only_nvidia = len(sys.argv) > 1 and sys.argv[1] == "nvidia-only"
    if only_nvidia:
        # Restrict libglvnd to the NVIDIA ICD so Mesa contributes no devices at all.
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    from mujoco.egl import egl_ext as EGL
    devices = EGL.eglQueryDevicesEXT()
    print(f"NDEV {len(devices)}")
    for i, dev in enumerate(devices):
        try:
            disp = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, dev, None)
            if not disp or not EGL.eglInitialize(disp, None, None):
                print(f"DEV {i} no-display")
                continue
            vendor = EGL.eglQueryString(disp, EGL.EGL_VENDOR)
            ver = EGL.eglQueryString(disp, EGL.EGL_VERSION)
            dec = lambda b: b.decode() if hasattr(b, "decode") else str(b)
            print(f"DEV {i} vendor={dec(vendor)} version={dec(ver)}")
        except Exception as exc:
            # PyOpenGL's EGLError repr spans several lines and hides the one field that
            # matters. Pull out the EGL error enum and name it: EGL_BAD_ACCESS points at
            # permissions/cgroup, EGL_NOT_INITIALIZED/EGL_BAD_ALLOC at a driver that has
            # the device but cannot bring up graphics on it.
            code = getattr(exc, "err", None)
            names = {
                0x3000: "EGL_SUCCESS", 0x3001: "EGL_NOT_INITIALIZED", 0x3002: "EGL_BAD_ACCESS",
                0x3003: "EGL_BAD_ALLOC", 0x3004: "EGL_BAD_ATTRIBUTE", 0x3005: "EGL_BAD_CONFIG",
                0x3006: "EGL_BAD_CONTEXT", 0x3007: "EGL_BAD_CURRENT_SURFACE",
                0x3008: "EGL_BAD_DISPLAY", 0x3009: "EGL_BAD_MATCH", 0x300A: "EGL_BAD_NATIVE_PIXMAP",
                0x300B: "EGL_BAD_NATIVE_WINDOW", 0x300C: "EGL_BAD_PARAMETER",
                0x300D: "EGL_BAD_SURFACE", 0x300E: "EGL_CONTEXT_LOST",
            }
            label = names.get(code, hex(code) if isinstance(code, int) else "?")
            msg = " ".join(str(exc).split())[:120]
            print(f"DEV {i} error={type(exc).__name__} egl_error={label} detail={msg}")
    """
)


_OSMESA_PROBE = textwrap.dedent(
    """
    import os, sys
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["MUJOCO_GL"] = "osmesa"
    try:
        from robosuite.renderers.context.osmesa_context import OSMesaGLContext
        OSMesaGLContext(max_width=64, max_height=64, device_id=-1)
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}".replace("\\n", " ")[:200])
        sys.exit(1)
    print("OK")
    """
)


# eglQueryDevicesEXT() is the FIRST thing robosuite calls, so if it fails nothing about
# device indices matters. When it fails, the useful facts are (1) the exception itself,
# (2) which libEGL actually got loaded -- a Mesa or bundled libEGL shadowing the system
# libglvnd one will not export the EXT_device_* entry points -- and (3) the EGL *client*
# extensions, which name the extensions robosuite depends on.
_EGL_DIAG = textwrap.dedent(
    """
    import os, sys
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # MUST match robosuite: egl_context.py does `from mujoco.egl import egl_ext as EGL`,
    # NOT `from OpenGL import EGL`. The PyOpenGL namespace does not export the EXT entry
    # points, so querying it reports a spurious failure that has nothing to do with the
    # driver -- which is exactly the false alarm an earlier version of this script raised.
    try:
        from mujoco.egl import egl_ext as EGL
    except Exception as exc:
        print(f"IMPORT_ERR {type(exc).__name__}: {exc}")
        sys.exit(0)

    # Which shared objects are actually mapped for EGL?
    try:
        maps = open("/proc/self/maps").read().splitlines()
        libs = sorted({ln.split()[-1] for ln in maps
                       if "libEGL" in ln.split()[-1] or "libGLdispatch" in ln.split()[-1]})
        for lib in libs:
            print(f"LOADED {lib}")
    except Exception:
        pass

    # Client extensions (queried on EGL_NO_DISPLAY). robosuite needs EXT_device_enumeration
    # (for eglQueryDevicesEXT) and EXT_platform_device (for eglGetPlatformDisplayEXT).
    try:
        from OpenGL import EGL as _PYEGL
        exts = _PYEGL.eglQueryString(_PYEGL.EGL_NO_DISPLAY, _PYEGL.EGL_EXTENSIONS)
        exts = exts.decode() if hasattr(exts, "decode") else str(exts)
        print(f"CLIENT_EXTS {exts}")
    except Exception as exc:
        print(f"CLIENT_EXTS_ERR {type(exc).__name__}: {exc}")

    try:
        print(f"COUNT {len(EGL.eglQueryDevicesEXT())}")
    except Exception as exc:
        print(f"QUERY_ERR {type(exc).__name__}: {exc}")
    """
)


def count_egl_devices(log) -> int | None:
    """Number of devices eglQueryDevicesEXT reports, logging WHY when it fails."""
    out = subprocess.run([sys.executable, "-c", _EGL_DIAG], capture_output=True, text=True)
    count = None
    for line in (out.stdout or "").strip().splitlines():
        if line.startswith("COUNT "):
            count = int(line.split()[1])
        elif line.startswith("LOADED "):
            log(f"  loaded EGL lib: {line[7:]}")
        elif line.startswith("CLIENT_EXTS "):
            exts = line[len("CLIENT_EXTS "):].split()
            needed = ["EGL_EXT_device_enumeration", "EGL_EXT_device_query", "EGL_EXT_platform_device"]
            log(f"  EGL client extensions ({len(exts)}): {' '.join(exts) if exts else '<none>'}")
            for ext in needed:
                mark = "present" if ext in exts else "MISSING  <- robosuite needs this"
                log(f"    {ext}: {mark}")
        elif line.split()[0].endswith("_ERR") or line.startswith("QUERY_ERR"):
            log(f"  {line}")
    if (out.stderr or "").strip() and count is None:
        log(f"  stderr: {out.stderr.strip().splitlines()[-1][:200]}")
    return count


def probe(script: str, *args: str) -> tuple[bool, str]:
    cmd = [sys.executable, "-c", script] + [a for a in args if a is not None]
    res = subprocess.run(cmd, capture_output=True, text=True)
    detail = res.stdout.strip() or (res.stderr.strip().splitlines() or [""])[-1]
    return res.returncode == 0, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--emit-exports",
        action="store_true",
        help="Print only the export lines for a working backend (for `eval $(...)`).",
    )
    args = ap.parse_args()
    log = (lambda *a: None) if args.emit_exports else print

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    idx_used = int(cvd) if (cvd or "").isdigit() else 0
    log(f"CUDA_VISIBLE_DEVICES = {cvd!r}   <- robosuite uses this as an EGL LIST INDEX")
    log(f"  => robosuite would request EGL device index {idx_used} (0 when unset)")
    log(f"MUJOCO_EGL_DEVICE_ID = {os.environ.get('MUJOCO_EGL_DEVICE_ID')!r}")
    for d in ("/usr/share/glvnd/egl_vendor.d",):
        if os.path.isdir(d):
            log(f"EGL vendor ICDs in {d}: {sorted(os.listdir(d))}")

    # These are what actually separate "wrong device index" from "NVIDIA EGL vendor not
    # usable in this job at all". Without a GPU the NVIDIA vendor enumerates NOTHING,
    # however early its ICD sorts -- so a CPU-node run says nothing about GPU nodes.
    import ctypes.util
    import glob

    # MIG is a hard stop, not a tuning problem: the Alliance docs (Multi-Instance GPU.md)
    # state "Graphic APIs are not supported (for example, OpenGL, Vulkan, etc.)". On fir
    # roughly half the GPU nodes are MIG-configured, and a full GPU there must be requested
    # as --gpus=h100:1 (NOT the --gres=gpu:h100:1 spelling that nibi uses).
    mig = None
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        if out.strip():
            log("nvidia-smi -L:")
            for line in out.strip().splitlines():
                log(f"    {line}")
            mig = "MIG" in out
    except Exception:
        pass
    if mig:
        log("")
        log("*** THIS IS A MIG INSTANCE. MIG does not support graphics APIs (OpenGL/Vulkan/EGL),")
        log("*** so EGL rendering CANNOT work here regardless of device index or driver.")
        log("*** Request a FULL GPU instead -- on fir that is:  --gpus=h100:1")
        log("")

    nvidia_nodes = sorted(glob.glob("/dev/nvidia*"))
    dri_nodes = sorted(os.path.basename(x) for x in glob.glob("/dev/dri/*"))
    libegl_nvidia = ctypes.util.find_library("EGL_nvidia")
    log(f"/dev/nvidia*: {nvidia_nodes or 'NONE'}")
    log(f"/dev/dri:     {dri_nodes or 'NONE'}")
    # EGL on NVIDIA needs to open the DRM render node. If the job's cgroup or the node's
    # permissions deny that, eglInitialize fails even though CUDA works fine.
    for node in sorted(glob.glob("/dev/dri/renderD*")) + sorted(glob.glob("/dev/dri/card*")):
        try:
            st = os.stat(node)
            log(f"    {node}: mode={oct(st.st_mode)[-3:]} "
                f"readable={os.access(node, os.R_OK)} writable={os.access(node, os.W_OK)}")
        except Exception as exc:
            log(f"    {node}: stat failed ({exc})")
    log(f"libEGL_nvidia.so.0 resolves: {bool(libegl_nvidia)}"
        f"{'' if libegl_nvidia else '  <- NVIDIA EGL vendor library not loadable here'}")

    on_gpu_node = bool(nvidia_nodes)
    if not on_gpu_node:
        log("")
        log("*** THIS NODE HAS NO GPU. The NVIDIA EGL vendor can enumerate no devices, so")
        log("*** the results below describe software rendering only and do NOT tell you")
        log("*** why an EGL job failed on a GPU node. Re-run this inside a --gres=gpu job.")
        log("")

    log("EGL client diagnostics:")
    n = count_egl_devices(log)

    # Map every EGL device, ignoring robosuite's CUDA_VISIBLE_DEVICES restriction.
    for label, arg in (("all vendors", None), ("NVIDIA ICD only", "nvidia-only")):
        res = subprocess.run(
            [sys.executable, "-c", _ENUM_PROBE] + ([arg] if arg else []),
            capture_output=True, text=True,
        )
        lines = [ln for ln in (res.stdout or "").strip().splitlines() if ln.startswith(("NDEV", "DEV"))]
        if not lines:
            continue
        log(f"  EGL devices ({label}):")
        for ln in lines:
            log(f"    {ln}")

    log(f"eglQueryDevicesEXT() reports {n if n is not None else 'ERROR (see above)'} device(s)")

    working: list[int] = []
    # Never let an unknown count skip the probe: the probe is the ground truth, the count
    # is only a convenience. Fall back to a few indices so a counting failure cannot
    # silently downgrade us to software rendering.
    indices = range(n) if n else range(max(1, min(4, len(nvidia_nodes) or 1)))
    if not n:
        log(f"  (device count unknown -- probing indices {list(indices)} anyway)")
    if True:
        for idx in indices:
            ok, detail = probe(_PROBE, str(idx))
            # Show the vendor on success too: a device can initialize via a Mesa software
            # driver, which "works" but renders on CPU. NVIDIA in the vendor string is the
            # difference between GPU-accelerated eval and a silent slow path.
            log(f"  EGL device {idx}: {detail}")
            if ok:
                working.append((idx, detail))

    nvidia_first = [i for i, d in working if "nvidia" in d.lower()]
    if working:
        idxs = [i for i, _ in working]
        log(f"\nWorking EGL device indices: {idxs}")
        if cvd is not None and cvd.isdigit() and int(cvd) not in idxs:
            log(f"   NOTE: CUDA_VISIBLE_DEVICES={cvd} is NOT a working EGL index. robosuite "
                f"asserts MUJOCO_EGL_DEVICE_ID is a substring of CUDA_VISIBLE_DEVICES, so it "
                f"can only ever try index {idx_used} -- see the NVIDIA-only retry below.")
    if nvidia_first:
        chosen = nvidia_first[0]
        print("export MUJOCO_GL=egl")
        print("export PYOPENGL_PLATFORM=egl")
        print(f"export MUJOCO_EGL_DEVICE_ID={chosen}")
        return 0
    if working:
        log("   No working device reported an NVIDIA vendor -- that means GPU-accelerated")
        log("   rendering is NOT what you would get. Trying the NVIDIA-only ICD before")
        log("   settling for software rendering.")

    # Before giving up on GPU rendering: robosuite can only try the index named by
    # CUDA_VISIBLE_DEVICES, so if Mesa devices sort ahead of the NVIDIA ones that index is
    # a software device and EGL fails. Pinning libglvnd to the NVIDIA ICD removes the Mesa
    # devices from the list entirely, which renumbers the GPU to index 0.
    idx = str(idx_used)
    ok, detail = probe(_PROBE, idx, "nvidia-only")
    log(f"\nRetry with NVIDIA-only EGL ICD (index {idx}): {detail}")
    if ok:
        log("   => Mesa EGL devices were shadowing the GPU in the device list.")
        print("export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
        print("export MUJOCO_GL=egl")
        print("export PYOPENGL_PLATFORM=egl")
        print(f"export MUJOCO_EGL_DEVICE_ID={idx}")
        return 0

    ok, detail = probe(_OSMESA_PROBE)
    if ok:
        if mig:
            log("\nNo EGL device works -- this is a MIG instance, where graphics APIs are")
            log("unsupported by design. Re-run on a FULL GPU (fir: --gpus=h100:1) before")
            log("concluding anything about this cluster's EGL support.")
        elif on_gpu_node:
            log("\nNo EGL device works on this GPU node. OSMesa (software rendering) does.")
        else:
            log("\nOSMesa (software rendering) works. Expected on a CPU node; not a GPU-node answer.")
        log("WARNING: OSMesa renders on CPU -- eval rollouts will be much slower than on nibi.")
        print("export MUJOCO_GL=osmesa")
        print("export PYOPENGL_PLATFORM=osmesa")
        return 0

    log(f"\nNeither EGL nor OSMesa works. OSMesa said: {detail}")
    if mig:
        log("Almost certainly because this is a MIG instance -- retry on a full GPU.")
    log("Ask the cluster's support desk which headless GL backend their GPU nodes provide.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
