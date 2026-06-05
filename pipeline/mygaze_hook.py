"""
MyGaze SDK Hook
---------------
Drop-in replacement for the MyGazeHook stub in cognitive_strain_detector.py.
Uses ctypes to call myGazeAPI64.dll directly — no iViewNG-Server needed.

Usage: copy this file into your pipeline/ folder and replace the MyGazeHook
class in cognitive_strain_detector.py with this one.
"""

import ctypes
import ctypes.wintypes
import time
import os


# ── DLL path ──────────────────────────────────────────────────────────────────
MYGAZE_DLL = r"C:\Program Files (x86)\Visual Interaction\myGaze SDK\bin\myGazeAPI64.dll"

# ── Return codes from the myGaze API ─────────────────────────────────────────
RET_SUCCESS          = 1
RET_NO_VALID_DATA    = 2
RET_CALIBRATION_ABORTED = 3
RET_SERVER_IS_RUNNING   = 4
RET_EYETRACKING_APPLICATION_NOT_RUNNING = 5
RET_WRONG_PARAMETER  = 6


# ── Gaze data struct (matches iV_SampleStruct in myGaze SDK) ─────────────────
class IVSample(ctypes.Structure):
    """
    Mirrors the SampleStruct from the myGaze/iViewX API.
    Fields: timestamp (ms), gaze x/y for left and right eye.
    """
    _fields_ = [
        ("timestamp",       ctypes.c_longlong),
        ("leftEye_gazeX",   ctypes.c_double),
        ("leftEye_gazeY",   ctypes.c_double),
        ("leftEye_diam",    ctypes.c_double),
        ("rightEye_gazeX",  ctypes.c_double),
        ("rightEye_gazeY",  ctypes.c_double),
        ("rightEye_diam",   ctypes.c_double),
    ]


class MyGazeHook:
    """
    Real myGaze SDK hook using ctypes.
    Replaces the stub in cognitive_strain_detector.py.

    Calling sequence:
        hook = MyGazeHook()
        hook.connect()          # connects + auto-calibrates
        pt = hook.get_gaze_point()   # returns (x, y) or None
        hook.disconnect()
    """

    def __init__(self, dll_path: str = MYGAZE_DLL):
        self._connected = False
        self._dll = None
        self._dll_path = dll_path

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        if not os.path.exists(self._dll_path):
            print(f"[MyGaze] DLL not found at: {self._dll_path}")
            print("[MyGaze] Falling back to stub mode")
            return

        try:
            self._dll = ctypes.CDLL(self._dll_path)
            self._setup_prototypes()
        except OSError as e:
            print(f"[MyGaze] Failed to load DLL: {e}")
            return

        # Connect to the eye tracker (device must be plugged in)
        ret = self._dll.iV_Connect()
        if ret == RET_SUCCESS:
            print("[MyGaze] Connected successfully")
            self._connected = True
            self._start_tracking()
        elif ret == RET_SERVER_IS_RUNNING:
            print("[MyGaze] Already connected")
            self._connected = True
            self._start_tracking()
        else:
            print(f"[MyGaze] Connection failed (code {ret}). "
                  "Is the device plugged in?")

    def _setup_prototypes(self):
        """Tell ctypes the return types for each API function."""
        for fn in ['iV_Connect', 'iV_Disconnect',
                   'iV_StartRecording', 'iV_StopRecording',
                   'iV_GetSample', 'iV_Calibrate']:
            if hasattr(self._dll, fn):
                getattr(self._dll, fn).restype = ctypes.c_int

        # iV_GetSample takes a pointer to IVSample
        if hasattr(self._dll, 'iV_GetSample'):
            self._dll.iV_GetSample.argtypes = [ctypes.POINTER(IVSample)]

    def _start_tracking(self):
        """Run calibration then start recording."""
        print("[MyGaze] Starting calibration — follow the dot on screen...")
        ret = self._dll.iV_Calibrate()
        if ret == RET_SUCCESS:
            print("[MyGaze] Calibration complete")
        else:
            print(f"[MyGaze] Calibration returned code {ret} — continuing anyway")

        ret = self._dll.iV_StartRecording()
        if ret == RET_SUCCESS:
            print("[MyGaze] Recording started")
        else:
            print(f"[MyGaze] StartRecording returned code {ret}")

    # ── Gaze data ─────────────────────────────────────────────────────────────

    def get_gaze_point(self):
        """
        Returns (x, y) gaze position in screen pixels, or None if unavailable.
        Averages left and right eye for a single gaze point.
        """
        if not self._connected or self._dll is None:
            return None

        sample = IVSample()
        ret = self._dll.iV_GetSample(ctypes.byref(sample))

        if ret != RET_SUCCESS:
            return None

        # Average both eyes; fall back to whichever eye has valid data
        lx, ly = sample.leftEye_gazeX,  sample.leftEye_gazeY
        rx, ry = sample.rightEye_gazeX, sample.rightEye_gazeY

        # Filter out zeroes (eye not detected)
        valid = [(x, y) for x, y in [(lx, ly), (rx, ry)] if x != 0 or y != 0]
        if not valid:
            return None

        x = sum(v[0] for v in valid) / len(valid)
        y = sum(v[1] for v in valid) / len(valid)
        return (x, y)

    def get_pupil_diameter(self):
        """
        Returns average pupil diameter (proxy for cognitive load).
        Larger pupil = more cognitive effort.
        """
        if not self._connected or self._dll is None:
            return None

        sample = IVSample()
        ret = self._dll.iV_GetSample(ctypes.byref(sample))
        if ret != RET_SUCCESS:
            return None

        diams = [d for d in [sample.leftEye_diam, sample.rightEye_diam] if d > 0]
        return sum(diams) / len(diams) if diams else None

    # ── Disconnection ─────────────────────────────────────────────────────────

    def disconnect(self):
        if self._connected and self._dll is not None:
            self._dll.iV_StopRecording()
            self._dll.iV_Disconnect()
            print("[MyGaze] Disconnected")
        self._connected = False