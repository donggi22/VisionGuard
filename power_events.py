"""
Windows 절전/빠른 시작 종료 감지.

숨겨진 창을 만들어 WM_POWERBROADCAST(PBT_APMSUSPEND) 메시지로 절전 진입을 감지한다.
빠른 시작 종료는 프로세스를 끝내지 않고 최대 절전으로 멈추므로 절전과 같은 경로로 온다.
(완전 종료/재부팅은 WM_ENDSESSION으로 시도했으나 작업 스케줄러로 실행된 프로세스에는
 알림이 오지 않아 제외했다)
"""

import ctypes
import threading
from ctypes import wintypes
from typing import Callable

WM_POWERBROADCAST = 0x0218
PBT_APMSUSPEND = 0x0004
DEVICE_NOTIFY_WINDOW_HANDLE = 0

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.DefWindowProcW.restype = LRESULT
_user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
_user32.RegisterClassW.restype = wintypes.ATOM
_user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
]
_user32.CreateWindowExW.restype = wintypes.HWND
_user32.RegisterSuspendResumeNotification.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_user32.RegisterSuspendResumeNotification.restype = wintypes.HANDLE
_user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
_user32.GetMessageW.restype = wintypes.BOOL
_user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
_user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
_user32.DispatchMessageW.restype = LRESULT
_kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE


def start_power_watcher(on_suspend: Callable[[], None]) -> threading.Thread:
    """
    숨겨진 창과 메시지 루프를 백그라운드 스레드로 띄운다.
    콜백은 메시지 처리 중에 동기 실행되며, Windows는 처리가 끝날 때까지 절전을
    잠시(약 2초) 기다리므로 콜백 안에서 알림 전송을 끝내야 한다.
    """

    def _wndproc(hwnd, msg, wparam, lparam):
        try:
            if msg == WM_POWERBROADCAST:
                if wparam == PBT_APMSUSPEND:
                    on_suspend()
                return 1
        except Exception as e:
            # 콜백 예외가 ctypes 경계를 넘으면 메시지 루프가 망가지므로 여기서 막는다
            print(f"[전원] 이벤트 처리 오류: {e}")
        return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _run():
        wndproc = WNDPROC(_wndproc)  # 스레드가 살아있는 동안 참조를 유지해야 GC되지 않는다
        hinst = _kernel32.GetModuleHandleW(None)
        wc = WNDCLASSW()
        wc.lpfnWndProc = wndproc
        wc.hInstance = hinst
        wc.lpszClassName = "VisionGuardPowerWatcher"
        if not _user32.RegisterClassW(ctypes.byref(wc)):
            print(f"[전원] 창 클래스 등록 실패: {ctypes.get_last_error()}")
            return

        # 전원 브로드캐스트는 최상위 창에만 오므로 메시지 전용 창(HWND_MESSAGE)이 아닌 숨김 최상위 창을 만든다
        hwnd = _user32.CreateWindowExW(
            0, wc.lpszClassName, "VisionGuard", 0, 0, 0, 0, 0, None, None, hinst, None
        )
        if not hwnd:
            print(f"[전원] 감지 창 생성 실패: {ctypes.get_last_error()}")
            return

        # 비대화형 세션(작업 스케줄러)에서는 브로드캐스트를 못 받을 수 있어 절전 알림을 직접 등록한다
        if not _user32.RegisterSuspendResumeNotification(hwnd, DEVICE_NOTIFY_WINDOW_HANDLE):
            print(f"[전원] 절전 알림 등록 실패: {ctypes.get_last_error()}")

        msg = wintypes.MSG()
        while _user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))

    t = threading.Thread(target=_run, daemon=True, name="power-watcher")
    t.start()
    return t
