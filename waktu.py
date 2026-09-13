"""
waktu.py
========
Satu sumber kebenaran WAKTU untuk seluruh proyek: zona WIB (Asia/Jakarta).

Kenapa modul terpisah? Supaya perilaku tidak bergantung pada timezone server.
Server ini disetel ke Asia/Jakarta, tetapi kode tetap eksplisit WIB agar aman
bila timezone server berubah (mis. VPS baru yang masih UTC).

Cara pakai:
    from waktu import now_wib, today_wib
    print(now_wib().strftime("%Y-%m-%d %H:%M:%S"))   # waktu WIB
    print(today_wib())                                # tanggal WIB
"""

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    WIB = ZoneInfo("Asia/Jakarta")
except Exception:  # tzdata tidak tersedia -> pakai offset tetap (+07:00)
    WIB = timezone(timedelta(hours=7), "WIB")


def now_wib():
    """Waktu sekarang di zona WIB (datetime aware)."""
    return datetime.now(WIB)


def today_wib():
    """Tanggal hari ini menurut WIB."""
    return now_wib().date()


def to_wib(dt):
    """Konversi datetime (naive dianggap UTC, atau aware) -> WIB aware."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(WIB)


def fmt_wib(dt=None, fmt="%Y-%m-%d %H:%M:%S"):
    """Format waktu (default: sekarang WIB)."""
    return (dt or now_wib()).strftime(fmt)
