import ctypes, os, sys, time, threading, traceback, math, random, json, struct
from ctypes import wintypes
from dataclasses import dataclass, field, asdict

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import numpy as np
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QCheckBox, QSlider, QPushButton, QFrame)

user32 = ctypes.windll.user32


# ---------------------------------------------------------------- config
def _base():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(_base(), "config.json")


@dataclass
class Config:
    monitor: int = 1
    region: list | None = None
    fps: int = 60
    mode: str = "auto"
    model: str = "models/rivals.pt"
    conf: float = 0.35
    imgsz: int = 640
    device: str = "0"
    color_ranges: list = field(default_factory=list)
    ignore_ranges: list = field(default_factory=list)
    min_area: int = 90
    max_area: int = 400000
    min_aspect: float = 1.3
    detect_scale: float = 0.5
    detect_fps: int = 60
    aim_enabled: bool = True
    aim_key: int = 0x05
    fov: int = 200
    smoothing: float = 0.28
    max_step: float = 60.0
    prediction: float = 0.0
    jitter: float = 0.6
    head_bias: float = 0.22
    trigger_enabled: bool = False
    trigger_key: int = 0x06
    trigger_fov: int = 14
    trigger_delay_ms: int = 25
    trigger_hold_ms: int = 18
    overlay: bool = True
    box_color: str = "#FF00FF"
    fov_color: str = "#00FF88"
    require_focus: bool = True
    window_title: str = "Roblox"
    quit_key: int = 0x23
    mem_esp: bool = False
    cam_fov: float = 70.0
    off_name: int = 0x28
    off_children: int = 0x40
    off_pos: int = 0x24


def load_cfg() -> Config:
    c = Config()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                for k, v in json.load(f).items():
                    if hasattr(c, k):
                        setattr(c, k, v)
        except Exception:
            pass
    return c


def save_cfg(c: Config) -> None:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(asdict(c), f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------- mouse
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
INPUT_MOUSE = 0


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _U(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


_acc_x = 0.0
_acc_y = 0.0


def _send(dx, dy, flags):
    inp = INPUT(INPUT_MOUSE, _U(MOUSEINPUT(dx, dy, 0, flags, 0, None)))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def mouse_move(dx, dy):
    global _acc_x, _acc_y
    _acc_x += dx
    _acc_y += dy
    ix, iy = int(_acc_x), int(_acc_y)
    if ix == 0 and iy == 0:
        return
    _acc_x -= ix
    _acc_y -= iy
    _send(ix, iy, MOUSEEVENTF_MOVE)


def left_down():
    _send(0, 0, MOUSEEVENTF_LEFTDOWN)


def left_up():
    _send(0, 0, MOUSEEVENTF_LEFTUP)


def key_down(vk):
    return user32.GetAsyncKeyState(vk) & 0x8000 != 0


def fg_title():
    b = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(user32.GetForegroundWindow(), b, 256)
    return b.value


# ---------------------------------------------------------------- capture
class Capture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = None
        self._cam = None
        self._sct = None
        if cfg.region is None:
            import mss
            with mss.mss() as sct:
                m = sct.monitors[cfg.monitor]
            self.region = [m["left"], m["top"], m["width"], m["height"]]
        else:
            self.region = list(cfg.region)
        l, t, w, h = self.region
        try:
            import dxcam
            self._cam = dxcam.create(output_idx=cfg.monitor - 1,
                                     region=(l, t, l + w, t + h),
                                     target_fps=cfg.detect_fps)
            self._cam.grab()
            self.backend = "dxcam"
            return
        except Exception:
            pass
        import mss
        self._sct = mss.mss()
        self.backend = "mss"

    def grab(self):
        l, t, w, h = self.region
        if self.backend == "dxcam":
            return self._cam.grab()
        raw = self._sct.grab({"left": l, "top": t, "width": w, "height": h})
        return np.asarray(raw)[:, :, :3][:, :, ::-1]


# ---------------------------------------------------------------- detect
class Detection:
    __slots__ = ("x1", "y1", "x2", "y2", "conf")

    def __init__(self, x1, y1, x2, y2, conf=1.0):
        self.x1, self.y1, self.x2, self.y2, self.conf = x1, y1, x2, y2, conf

    @property
    def cx(self):
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self):
        return (self.y1 + self.y2) * 0.5

    def aim_point(self, bias):
        return self.cx, self.y1 + (self.y2 - self.y1) * bias


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kind = "color"
        self.model = None
        if os.path.exists(cfg.model):
            try:
                from ultralytics import YOLO
                self.model = YOLO(cfg.model)
                self.kind = "yolo"
            except Exception:
                pass

    def detect(self, frame):
        if self.kind == "yolo":
            r = self.model.predict(source=frame, conf=self.cfg.conf,
                                   imgsz=self.cfg.imgsz, device=self.cfg.device,
                                   verbose=False)[0]
            return [Detection(*b.xyxy[0].tolist(), float(b.conf[0]))
                    for b in r.boxes]

        sc = self.cfg.detect_scale
        small = cv2.resize(frame, None, fx=sc, fy=sc,
                           interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        mask = None
        for h1, s1, v1, h2, s2, v2 in self.cfg.color_ranges:
            m = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        if mask is None:
            return []
        for h1, s1, v1, h2, s2, v2 in self.cfg.ignore_ranges:
            m = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(m))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        out = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / sc
        amin, amax = self.cfg.min_area * sc * sc, self.cfg.max_area * sc * sc
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if not (amin <= w * h <= amax):
                continue
            if h <= 2 or w <= 2:
                continue
            if h < w * self.cfg.min_aspect:
                continue
            out.append(Detection(x * inv, y * inv, (x + w) * inv, (y + h) * inv))
        return out


# ---------------------------------------------------------------- aim
def pick_target(dets, cx, cy, radius, bias):
    best, best_d = None, radius
    for d in dets:
        px, py = d.aim_point(bias)
        dist = math.hypot(px - cx, py - cy)
        if dist < best_d:
            best, best_d = d, dist
    return best


def aim_step(dets, cx, cy, cfg, prev):
    if not dets:
        return 0.0, 0.0
    t = pick_target(dets, cx, cy, cfg.fov, cfg.head_bias)
    if t is None:
        return 0.0, 0.0
    px, py = t.aim_point(cfg.head_bias)
    key = (round(t.cx / 24), round(t.cy / 24))
    vx = vy = 0.0
    if key in prev:
        ox, oy = prev[key]
        vx, vy = px - ox, py - oy
    prev.clear()
    prev[key] = (px, py)
    if cfg.prediction:
        px += vx * cfg.prediction
        py += vy * cfg.prediction
    dx, dy = (px - cx) * cfg.smoothing, (py - cy) * cfg.smoothing
    mag = math.hypot(dx, dy)
    if mag > cfg.max_step:
        s = cfg.max_step / mag
        dx, dy = dx * s, dy * s
    if cfg.jitter:
        dx += random.uniform(-cfg.jitter, cfg.jitter)
        dy += random.uniform(-cfg.jitter, cfg.jitter)
    return dx, dy


# ---------------------------------------------------------------- memory
class MemESP:
    def __init__(self, cfg):
        import pymem, pymem.process
        self.cfg = cfg
        self.pm = pymem.Pymem("RobloxPlayerBeta.exe")
        mod = pymem.process.module_from_name(self.pm.process_handle,
                                             "RobloxPlayerBeta.exe")
        self.base = mod.lpBaseOfDll
        self.size = mod.SizeOfImage
        self.dm = self._find_dm()
        if not self.dm:
            raise RuntimeError("DataModel not resolved")

    def _find_dm(self):
        pat = b"\x48\x8B\x05\x00\x00\x00\x00\x48\x8B\x88\x00\x00\x00\x00\x48\x8B\x01"
        buf = self.pm.read_bytes(self.base, self.size)
        i = 0
        while True:
            i = buf.find(pat[:3], i)
            if i < 0:
                return None
            try:
                rip = struct.unpack_from("<i", buf, i + 3)[0]
                p = self.pm.read_longlong(self.base + i + 7 + rip)
                if p and self._class(p) == "DataModel":
                    return p
            except Exception:
                pass
            i += 1

    def _str(self, addr):
        try:
            n = self.pm.read_int(addr + 0x10)
            n = n if 0 < n < 128 else 64
            return self.pm.read_bytes(addr, n).split(b"\x00")[0].decode("utf-8", "ignore")
        except Exception:
            return ""

    def _name(self, i):
        return self._str(i + self.cfg.off_name)

    def _class(self, i):
        try:
            return self._str(self.pm.read_longlong(i + 0x18) + 0x08)
        except Exception:
            return ""

    def _kids(self, i):
        try:
            b = self.pm.read_longlong(i + self.cfg.off_children)
            e = self.pm.read_longlong(i + self.cfg.off_children + 8)
            if not b or not e or e < b:
                return []
            n = min((e - b) // 8, 4000)
            return list(struct.unpack(f"<{n}Q", self.pm.read_bytes(b, n * 8)))
        except Exception:
            return []

    def _find(self, i, name, d=0):
        if d > 4:
            return None
        for c in self._kids(i):
            if self._name(c) == name:
                return c
            r = self._find(c, name, d + 1)
            if r:
                return r
        return None

    def _pos(self, part):
        try:
            cf = self.pm.read_longlong(part + 0xC8)
            o = self.cfg.off_pos
            return (self.pm.read_float(cf + o), self.pm.read_float(cf + o + 4),
                    self.pm.read_float(cf + o + 8))
        except Exception:
            return None

    def positions(self):
        pl = self._find(self.dm, "Players")
        if not pl:
            return []
        out = []
        for p in self._kids(pl):
            ch = self._find(p, "Character", 0)
            if not ch:
                continue
            hrp = self._find(ch, "HumanoidRootPart", 0)
            if hrp:
                v = self._pos(hrp)
                if v:
                    out.append(v)
        return out

    def project(self, world, vw, vh):
        ws = self._find(self.dm, "Workspace")
        cam = self._find(ws, "Camera") if ws else None
        if not cam:
            return None
        cf = self.pm.read_longlong(cam + 0xC8)
        o = self.cfg.off_pos
        cx, cy, cz = (self.pm.read_float(cf + o), self.pm.read_float(cf + o + 4),
                      self.pm.read_float(cf + o + 8))
        r = [self.pm.read_float(cf + k * 4) for k in range(9)]
        dx, dy, dz = world[0] - cx, world[1] - cy, world[2] - cz
        depth = -(dx * r[6] + dy * r[7] + dz * r[8])
        if depth < 0.5:
            return None
        th = math.tan(math.radians(self.cfg.cam_fov) * 0.5)
        sx = vw * 0.5 + (dx * r[0] + dy * r[1] + dz * r[2]) / depth * (vw * 0.5) / th
        sy = vh * 0.5 - (dx * r[3] + dy * r[4] + dz * r[5]) / depth * (vh * 0.5) / th
        return sx, sy, 1.0 / depth


# ---------------------------------------------------------------- overlay
class Overlay(QWidget):
    def __init__(self, region, shared):
        super().__init__()
        self.shared = shared
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.WindowTransparentForInput | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(*region)
        t = QTimer(self)
        t.timeout.connect(self.update)
        t.start(50)

    def paintEvent(self, _):
        s = self.shared
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        if s.get("show_boxes"):
            p.setPen(QPen(QColor(s.get("box_color", "#FF00FF")), 1))
            for d in list(s.get("dets", [])):
                p.drawRect(int(d.x1), int(d.y1),
                           int(d.x2 - d.x1), int(d.y2 - d.y1))
        cx, cy = s.get("center", (0, 0))
        f = s.get("fov", 0)
        if f:
            p.setPen(QPen(QColor(s.get("fov_color", "#00FF88")), 1))
            p.drawEllipse(int(cx - f), int(cy - f), f * 2, f * 2)
        p.setPen(QPen(QColor("#00FF88"), 1))
        p.drawLine(int(cx) - 6, int(cy), int(cx) + 6, int(cy))
        p.drawLine(int(cx), int(cy) - 6, int(cx), int(cy) + 6)
        p.end()


# ---------------------------------------------------------------- panel
CSS = """
QWidget#panel { background:#0e0f14; border:1px solid #23252f; border-radius:8px; }
QLabel { color:#c8ccd6; font-family:Consolas; font-size:11px; }
QLabel#title { color:#00ff88; font-size:12px; font-weight:bold; }
QLabel#status { color:#7f8494; font-size:10px; }
QLabel#val { color:#00ff88; }
QCheckBox { color:#c8ccd6; font-family:Consolas; font-size:11px; spacing:6px; }
QCheckBox::indicator { width:32px; height:16px; border-radius:8px;
    background:#2a2d38; border:1px solid #3a3e4c; }
QCheckBox::indicator:checked { background:#00ff88; border-color:#00ff88; }
QSlider::groove:horizontal { height:3px; background:#2a2d38; border-radius:2px; }
QSlider::handle:horizontal { width:11px; height:11px; margin:-5px 0;
    border-radius:5px; background:#00ff88; }
QPushButton { background:#1a1c24; color:#c8ccd6; border:1px solid #2a2d38;
    border-radius:4px; padding:4px; font-family:Consolas; font-size:10px; }
QPushButton:hover { border-color:#00ff88; color:#00ff88; }
QFrame#sep { background:#23252f; max-height:1px; }
"""


class Panel(QWidget):
    def __init__(self, cfg, shared, on_quit, on_mem):
        super().__init__()
        self.cfg, self.shared = cfg, shared
        self.on_quit, self.on_mem = on_quit, on_mem
        self._drag = None
        self.setObjectName("panel")
        self.setStyleSheet(CSS)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("RivalsAssist")
        self.setFixedWidth(232)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(5)

        head = QHBoxLayout()
        t = QLabel("RIVALS ASSIST")
        t.setObjectName("title")
        head.addWidget(t)
        head.addStretch()
        x = QPushButton("X")
        x.setFixedSize(20, 18)
        x.clicked.connect(on_quit)
        head.addWidget(x)
        lay.addLayout(head)
        lay.addWidget(self._sep())
        lay.addWidget(self._tg("Aimbot  [mouse5]", "aim_enabled"))
        lay.addWidget(self._tg("Triggerbot  [mouse4]", "trigger_enabled"))
        lay.addWidget(self._tg("ESP boxes  [F8]", "overlay"))
        lay.addWidget(self._tg("Memory ESP (walls)", "mem_esp", on_mem))
        lay.addWidget(self._tg("Only when focused", "require_focus"))
        lay.addWidget(self._sep())
        lay.addLayout(self._sl("FOV", "fov", 50, 500, 10))
        lay.addLayout(self._sl("Smoothing", "smoothing", 5, 90, 1, 100))
        lay.addLayout(self._sl("Max step", "max_step", 5, 200, 1))
        lay.addLayout(self._sl("Prediction", "prediction", 0, 200, 1, 100))
        lay.addLayout(self._sl("Jitter", "jitter", 0, 50, 1, 10))
        lay.addLayout(self._sl("Head bias", "head_bias", 0, 100, 1, 100))
        lay.addWidget(self._sep())
        lay.addLayout(self._sl("Trigger FOV", "trigger_fov", 2, 60, 1))
        lay.addLayout(self._sl("Trigger delay", "trigger_delay_ms", 0, 200, 1))
        lay.addLayout(self._sl("FPS cap", "fps", 30, 240, 10))
        lay.addWidget(self._sep())
        self.status = QLabel("booting...")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        sv = QPushButton("SAVE")
        sv.clicked.connect(self._save)
        row.addWidget(sv)
        hd = QPushButton("HIDE [INS]")
        hd.clicked.connect(self.hide)
        row.addWidget(hd)
        lay.addLayout(row)
        self.adjustSize()

    def _sep(self):
        f = QFrame()
        f.setObjectName("sep")
        f.setFixedHeight(1)
        return f

    def _tg(self, text, attr, hook=None):
        cb = QCheckBox(text)
        cb.setChecked(bool(getattr(self.cfg, attr)))
        cb.stateChanged.connect(lambda v, a=attr: setattr(self.cfg, a, bool(v)))
        if hook:
            cb.stateChanged.connect(hook)
        return cb

    def _sl(self, text, attr, lo, hi, step, div=1):
        box = QVBoxLayout()
        box.setSpacing(0)
        top = QHBoxLayout()
        top.addWidget(QLabel(text))
        top.addStretch()
        v = QLabel(f"{getattr(self.cfg, attr):.2f}" if div != 1
                   else f"{getattr(self.cfg, attr):.0f}")
        v.setObjectName("val")
        top.addWidget(v)
        box.addLayout(top)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setPageStep(step * 5)
        s.setValue(int(round(getattr(self.cfg, attr) * div)))
        s.valueChanged.connect(lambda val, a=attr, lb=v, d=div: self._slide(a, val, lb, d))
        box.addWidget(s)
        return box

    def _slide(self, attr, raw, label, div):
        val = raw / div
        setattr(self.cfg, attr, val)
        label.setText(f"{val:.2f}" if div != 1 else f"{val:.0f}")

    def _save(self):
        save_cfg(self.cfg)
        self.status.setText("config saved")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag and e.buttons() == Qt.LeftButton:
            self.move(e.globalPos() - self._drag)

    def mouseReleaseEvent(self, _):
        self._drag = None


# ---------------------------------------------------------------- main
def main():
    cfg = load_cfg()
    cap = Capture(cfg)
    det = Detector(cfg)
    w, h = cap.region[2], cap.region[3]
    cx, cy = w / 2, h / 2
    ox, oy = cap.region[0], cap.region[1]
    shared = {"dets": [], "center": (cx, cy), "fov": 0,
              "box_color": cfg.box_color, "fov_color": cfg.fov_color,
              "show_boxes": cfg.overlay, "panel_rect": None}
    app = QApplication(sys.argv)
    running = [True]
    mem = [None]

    def quit_all():
        running[0] = False
        save_cfg(cfg)
        app.quit()

    def mem_toggle(state):
        if not state:
            mem[0] = None
            return
        try:
            mem[0] = MemESP(cfg)
            panel.status.setText("memory ESP online")
        except Exception as e:
            mem[0] = None
            cfg.mem_esp = False
            panel.status.setText(f"mem failed: {e}")

    overlay = Overlay(cap.region, shared)
    overlay.show()
    panel = Panel(cfg, shared, quit_all, mem_toggle)
    panel.move(cap.region[0] + 24, cap.region[1] + 24)
    panel.show()

    prev, keys, last, last_status = {}, {}, [0.0], [0.0]

    def edge(vk):
        now = key_down(vk)
        was = keys.get(vk, False)
        keys[vk] = now
        return now and not was

    def loop():
        while running[0]:
            t0 = time.perf_counter()
            try:
                if edge(cfg.quit_key):
                    quit_all()
                    break
                if edge(0x2D):
                    panel.setVisible(not panel.isVisible())
                if edge(0x41):
                    cfg.overlay = not cfg.overlay
                if edge(0x42):
                    cfg.aim_enabled = not cfg.aim_enabled
                if edge(0x43):
                    cfg.trigger_enabled = not cfg.trigger_enabled

                if panel.isVisible():
                    g = panel.geometry()
                    shared["panel_rect"] = (g.x() - ox, g.y() - oy, g.width(), g.height())
                else:
                    shared["panel_rect"] = None

                frame = cap.grab()
                if frame is None:
                    time.sleep(1 / max(cfg.fps, 30))
                    continue

                dets = det.detect(frame)
                r = shared["panel_rect"]
                if r:
                    px, py, pw, ph = r
                    dets = [d for d in dets if not (px <= d.cx <= px + pw
                                                    and py <= d.cy <= py + ph)]

                if mem[0]:
                    try:
                        for pos in mem[0].positions():
                            pr = mem[0].project(pos, w, h)
                            if not pr:
                                continue
                            x, y, s = pr
                            if not (0 <= x <= w and 0 <= y <= h):
                                continue
                            hh = max(12, min(120, 90 * s))
                            dets.append(Detection(x - hh * 0.5, y - hh,
                                                  x + hh * 0.5, y))
                    except Exception:
                        pass

                shared["dets"] = dets
                shared["fov"] = cfg.fov if cfg.overlay else 0
                shared["show_boxes"] = cfg.overlay

                title = fg_title().lower()
                real = cfg.window_title.lower() in title
                focused = (not cfg.require_focus) or real or ("rivalsassist" in title)

                s_en, s_ig, s_rst = edge(0x76), edge(0x75), edge(0x74)
                if s_rst:
                    cfg.color_ranges, cfg.ignore_ranges = [], []
                    save_cfg(cfg)
                    panel.status.setText("HSV reset")
                if s_en or s_ig:
                    pt = wintypes.POINT()
                    user32.GetCursorPos(ctypes.byref(pt))
                    px_, py_ = pt.x - ox, pt.y - oy
                    if 0 <= px_ < w and 0 <= py_ < h:
                        b_, g_, r_ = frame[py_, px_]
                        hv = cv2.cvtColor(np.uint8([[[r_, g_, b_]]]),
                                          cv2.COLOR_RGB2HSV)[0][0]
                        hh_, ss_, vv_ = int(hv[0]), int(hv[1]), int(hv[2])
                        rng = [max(0, hh_ - 8), max(0, ss_ - 70), max(0, vv_ - 80),
                               min(179, hh_ + 8), 255, 255]
                        if s_ig:
                            cfg.ignore_ranges.append(rng)
                            panel.status.setText(f"IGNORE h{hh_} s{ss_} v{vv_}")
                        else:
                            cfg.color_ranges.append(rng)
                            panel.status.setText(f"ENEMY h{hh_} s{ss_} v{vv_}")
                        save_cfg(cfg)

                if focused:
                    if cfg.aim_enabled and key_down(cfg.aim_key):
                        dx, dy = aim_step(dets, cx, cy, cfg, prev)
                        if dx or dy:
                            mouse_move(dx, dy)
                    if real and cfg.trigger_enabled and key_down(cfg.trigger_key) \
                            and not key_down(0x01) \
                            and time.perf_counter() - last[0] > 0.06:
                        if pick_target(dets, cx, cy, cfg.trigger_fov, 0.5):
                            time.sleep(cfg.trigger_delay_ms / 1000)
                            left_down()
                            time.sleep(cfg.trigger_hold_ms / 1000)
                            left_up()
                            last[0] = time.perf_counter()

                if time.perf_counter() - last_status[0] > 0.25:
                    panel.status.setText(f"{det.kind} det:{len(dets)} "
                                         f"{'OK' if real else 'PANEL'}")
                    last_status[0] = time.perf_counter()
            except Exception:
                panel.status.setText("err: " + traceback.format_exc(limit=1).splitlines()[-1])
                time.sleep(0.2)
            el = time.perf_counter() - t0
            iv = 1 / max(cfg.fps, 30)
            if el < iv:
                time.sleep(iv - el)

    threading.Thread(target=loop, daemon=True).start()
    app.exec_()


if __name__ == "__main__":
    main()
