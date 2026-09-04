#!/usr/bin/env python
"""从 NLM 官方服务器批量下载 CHNCXR (Shenzhen Hospital CXR Set)。

官方没有提供打包归档，662 张影像是逐个文件服务的，因此需要脚本批量拉取。
本脚本只依赖标准库（urllib），不引入 requests 等额外依赖。

用法::

    # 先看要下什么、有多大，不实际下载
    python -m scripts.download_chncxr --dry-run

    # 下载影像（默认项）
    python -m scripts.download_chncxr

    # 影像 + 病灶标注 + 共识 ROI + 临床所见
    python -m scripts.download_chncxr --what images annotations roi clinical

    # 中断后重跑即自动续传；只想校验已有文件时：
    python -m scripts.download_chncxr --verify-only

下载完成后会在 ``data/splits/chncxr_files.sha256`` 写出指纹清单，
供实验记录追溯数据版本（见 data/README.md 第 8 节）。

设计要点：

* **文件清单来自服务器目录索引而非硬编码的命名规律。**
  虽然实测命名为 ``CHNCXR_0001_0.png`` ~ ``CHNCXR_0662_1.png``，
  但直接解析索引可以自动适应任何缺号或改名，且下载完能与服务器清单逐一对账。
* **断点续传。** 单张最大 7.8 MB、全集约 3.5--4 GB，中途断开是常态。
  已完整的文件跳过，未完成的用 HTTP Range 续传，避免重来。
* **完整性以字节数为准。** 每个文件下载后与服务器声明的 Content-Length 比对，
  不符即判为失败并重试——截断的 PNG 能打开却缺行，比下载失败更难发现。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = ("https://data.lhncbc.nlm.nih.gov/public/"
        "Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/")

# 各下载项：(相对路径, 目标子目录, 是否递归子目录)
TARGETS = {
    "images":      ("CXR_png/",         "CXR_png",         False),
    "annotations": ("Annotations/",     "Annotations",     True),
    "clinical":    ("ClinicalReadings/", "ClinicalReadings", False),
    "roi":         ("shenzhen_consensus_roi.csv", ".",      False),
    "readme":      ("NLM-ChinaCXRSet-ReadMe.docx", ".",     False),
}

# 官方规格，用于下载后核对
EXPECTED_IMAGES = 662
EXPECTED_NORMAL = 326       # CHNCXR_0001_0 ~ CHNCXR_0326_0
EXPECTED_ABNORMAL = 336     # CHNCXR_0327_1 ~ CHNCXR_0662_1

UA = "Mozilla/5.0 (compatible; XAI01-02-coursework/1.0)"
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _request(url: str, headers: dict | None = None, method: str = "GET"):
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    return urllib.request.Request(url, headers=h, method=method)


def fetch_bytes(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - 网络异常种类多，统一退避重试
            last = e
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"取回失败 {url}: {last}")


def remote_size(url: str, timeout: int = 30, retries: int = 3) -> int | None:
    """用 HEAD 取 Content-Length；服务器不支持则返回 None。"""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url, method="HEAD"), timeout=timeout) as r:
                n = r.headers.get("Content-Length")
                return int(n) if n is not None else None
        except Exception:  # noqa: BLE001
            time.sleep(1.0 * (2 ** attempt))
    return None


# 该服务器的索引是带大小列的表格，且 href 用单引号：
#   <td class='link'><a href='X.png'>X.png</a></td><td class='size'>5920202</td>
# 直接从表格取大小，可省掉逐个文件的 HEAD 请求（662 张就是 662 次往返）。
_ROW_RE = re.compile(
    r"""<td[^>]*class=['"]link['"][^>]*>\s*<a[^>]+href=['"]([^'"]+)['"][^>]*>.*?</a>\s*</td>"""
    r"""\s*<td[^>]*class=['"]size['"][^>]*>([^<]*)</td>""",
    re.I | re.S)
_HREF_RE = re.compile(r"""href=['"]([^'"]+)['"]""", re.I)


def _keep(href: str) -> bool:
    return not (href.startswith(("?", "/", "#")) or "://" in href
                or href.startswith("..") or href in ("index.html", "./"))


def list_index(dir_url: str) -> tuple[list[tuple[str, int | None]], list[str]]:
    """解析目录索引，返回 ([(文件名, 字节数或 None)], [子目录名])。"""
    html = fetch_bytes(urllib.parse.urljoin(dir_url, "index.html")).decode("utf-8", "replace")

    files: dict[str, int | None] = {}
    dirs: set[str] = set()

    for href, size in _ROW_RE.findall(html):
        if not _keep(href):
            continue
        name = urllib.parse.unquote(href)
        # 本服务器的子目录链接指向 SUBDIR/index.html，而非 SUBDIR/，
        # 不还原成目录就会把子目录当成待下载文件，递归也就失效了。
        if name.endswith("/index.html"):
            dirs.add(name[: -len("index.html")])
        elif name.endswith("/"):
            dirs.add(name)
        else:
            size = size.strip()
            files[name] = int(size) if size.isdigit() else None

    if not files and not dirs:                      # 表格结构不符时退回朴素解析
        for href in _HREF_RE.findall(html):
            if not _keep(href):
                continue
            name = urllib.parse.unquote(href)
            if name.endswith("/index.html"):
                dirs.add(name[: -len("index.html")])
            elif name.endswith("/"):
                dirs.add(name)
            else:
                files.setdefault(name, None)

    return sorted(files.items()), sorted(dirs)


def collect(rel: str, sub: str, recursive: bool) -> list[tuple[str, str, int | None]]:
    """展开一个下载项为 [(url, 相对目标路径, 已知字节数或 None)]。"""
    out: list[tuple[str, str, int | None]] = []
    sub = "" if sub == "." else sub

    if not rel.endswith("/"):                       # 单个文件：目录索引给不出大小，只能 HEAD
        url = urllib.parse.urljoin(BASE, rel)
        name = os.path.basename(rel)
        out.append((url, os.path.join(sub, name) if sub else name, remote_size(url)))
        return out

    stack = [(urllib.parse.urljoin(BASE, rel), sub)]
    while stack:
        url, dest = stack.pop()
        files, dirs = list_index(url)
        for name, size in files:
            out.append((urllib.parse.urljoin(url, urllib.parse.quote(name)),
                        os.path.join(dest, name) if dest else name, size))
        if recursive:
            for d in dirs:
                stack.append((urllib.parse.urljoin(url, urllib.parse.quote(d)),
                              os.path.join(dest, d.rstrip("/")) if dest else d.rstrip("/")))
    return out


def is_complete(path: str, expect: int | None) -> bool:
    """本地文件是否已完整。

    只判断「文件存在」是不够的：中断产生的截断文件同样存在，
    若据此跳过，损坏的文件会永远得不到修复，而截断的 PNG 打得开却缺行，
    比下载失败更难察觉。因此只要知道期望字节数就必须比对。
    """
    if not os.path.exists(path):
        return False
    return expect is None or os.path.getsize(path) == expect


def download_one(url: str, path: str, retries: int = 4, timeout: int = 120,
                 expect: int | None = None) -> str:
    """下载单个文件，支持续传。返回 'skip' / 'ok' / 'resume' / 'fail: ...'。

    expect 为索引页已给出的字节数；没有时才回退到 HEAD 查询。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if expect is None:
        expect = remote_size(url)

    if os.path.exists(path):
        have = os.path.getsize(path)
        if expect is None or have == expect:
            return "skip"
        if have > expect:                            # 本地反而更大，说明是坏文件
            os.remove(path)

    part = path + ".part"
    status = "ok"
    for attempt in range(retries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {}
            if have and expect is not None and have < expect:
                headers["Range"] = f"bytes={have}-"

            with urllib.request.urlopen(_request(url, headers), timeout=timeout) as r:
                # 请求了 Range 但服务器返回 200 = 不支持续传，只能从头写
                mode = "ab" if (r.status == 206 and have) else "wb"
                if mode == "wb":
                    have = 0
                elif have:
                    status = "resume"
                with open(part, mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)

            got = os.path.getsize(part)
            if expect is not None and got != expect:
                raise OSError(f"字节数不符：得到 {got}，应为 {expect}")

            os.replace(part, path)
            return status
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return f"fail: {e}"
            time.sleep(1.5 * (2 ** attempt))
    return "fail: 未知"


def sha256_of(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def check_images(img_dir: str) -> dict:
    """核对影像数量与正常/结核分布是否符合官方规格。"""
    if not os.path.isdir(img_dir):
        return {"present": 0, "ok": False, "note": "目录不存在"}
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    normal = [n for n in names if n.rsplit("_", 1)[-1].startswith("0")]
    abnormal = [n for n in names if n.rsplit("_", 1)[-1].startswith("1")]
    ok = (len(names) == EXPECTED_IMAGES
          and len(normal) == EXPECTED_NORMAL
          and len(abnormal) == EXPECTED_ABNORMAL)
    return {
        "present": len(names), "normal": len(normal), "abnormal": len(abnormal),
        "expected": EXPECTED_IMAGES, "ok": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="批量下载 CHNCXR (Shenzhen Hospital CXR Set)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.path.join("data", "raw", "chncxr"),
                    help="目标目录（默认 data/raw/chncxr）")
    ap.add_argument("--what", nargs="+", default=["images"], choices=sorted(TARGETS),
                    help="下载哪些内容（默认只下 images）")
    ap.add_argument("--workers", type=int, default=6,
                    help="并发数（默认 6；对公共服务器请勿调太高）")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true", help="只列出将要下载的文件与总大小")
    ap.add_argument("--verify-only", action="store_true", help="不下载，只校验本地已有文件")
    ap.add_argument("--manifest", default=os.path.join("data", "splits", "chncxr_files.sha256"),
                    help="指纹清单输出路径")
    args = ap.parse_args()

    img_dir = os.path.join(args.dest, "CXR_png")

    if args.verify_only:
        st = check_images(img_dir)
        print(f"影像目录 {img_dir}")
        print(f"  现有 {st['present']} 张"
              + (f"（正常 {st.get('normal')} / 结核 {st.get('abnormal')}）" if st['present'] else ""))
        print(f"  官方 {EXPECTED_IMAGES} 张（正常 {EXPECTED_NORMAL} / 结核 {EXPECTED_ABNORMAL}）")
        print("  结论: " + ("完整" if st["ok"] else "不完整，需继续下载"))
        return 0 if st["ok"] else 1

    print(f"来源: {BASE}")
    print(f"目标: {os.path.abspath(args.dest)}")
    print(f"内容: {', '.join(args.what)}\n")

    jobs: list[tuple[str, str, int | None]] = []
    for key in args.what:
        rel, sub, rec = TARGETS[key]
        print(f"  列举 {key} …", end=" ", flush=True)
        items = collect(rel, sub, rec)
        known = sum(1 for _, _, s in items if s)
        total = sum(s for _, _, s in items if s)
        print(f"{len(items)} 个文件"
              + (f"，索引已给出其中 {known} 个的大小，合计 {total/1024**3:.2f} GB" if known else ""))
        jobs.extend((u, os.path.join(args.dest, p), s) for u, p, s in items)

    # 去重（递归子目录时可能重复列到）
    seen, uniq = set(), []
    for u, p, s in jobs:
        if p not in seen:
            seen.add(p)
            uniq.append((u, p, s))
    jobs = uniq

    todo = [(u, p, s) for u, p, s in jobs if not is_complete(p, s)]
    broken = [p for _, p, s in jobs if os.path.exists(p) and not is_complete(p, s)]
    print(f"\n合计 {len(jobs)} 个文件，已完整 {len(jobs)-len(todo)} 个，待处理 {len(todo)} 个"
          + (f"（其中 {len(broken)} 个本地已存在但不完整，将重下或续传）" if broken else ""))

    if args.dry_run:
        known = [s for _, _, s in todo if s]
        if known:
            avg = sum(known) / len(known)
            est = sum(known) + avg * (len(todo) - len(known))
            print(f"\n待下载体积：约 {est/1024**3:.2f} GB"
                  f"（{len(known)}/{len(todo)} 个大小已知，均值 {avg/1024**2:.1f} MB）")
        print("\n前 10 个待下载文件：")
        for _, p, s in todo[:10]:
            print(f"    {p}" + (f"   {s/1024**2:.1f} MB" if s else ""))
        return 0

    if not todo:
        print("全部文件已完整，无需下载。")
    else:
        done = {"skip": 0, "ok": 0, "resume": 0}
        fails: list[tuple[str, str]] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download_one, u, p, args.retries, args.timeout, s): p
                    for u, p, s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                p = futs[fut]
                st = fut.result()
                if st.startswith("fail"):
                    fails.append((p, st))
                    log(f"  [{i}/{len(futs)}] 失败 {os.path.basename(p)} — {st}")
                else:
                    done[st] = done.get(st, 0) + 1
                    if i % 25 == 0 or i == len(futs):
                        el = time.time() - t0
                        rate = i / el if el else 0
                        eta = (len(futs) - i) / rate / 60 if rate else 0
                        log(f"  [{i}/{len(futs)}] 新下 {done['ok']} / 续传 {done['resume']} "
                            f"/ 失败 {len(fails)}   已用 {el/60:.1f} 分钟，约剩 {eta:.1f} 分钟")

        print(f"\n新下 {done['ok']}，续传 {done['resume']}，"
              f"已完整跳过 {len(jobs)-len(todo)}，失败 {len(fails)}")
        if fails:
            print("失败清单（重跑本命令即可续传）：")
            for p, st in fails[:20]:
                print(f"  {p}  {st}")
            if len(fails) > 20:
                print(f"  … 另有 {len(fails)-20} 个")

    # ---- 核对与指纹 ----
    if "images" in args.what:
        st = check_images(img_dir)
        print(f"\n影像核对：现有 {st['present']} / 应有 {EXPECTED_IMAGES}"
              + (f"（正常 {st.get('normal')}/{EXPECTED_NORMAL}，"
                 f"结核 {st.get('abnormal')}/{EXPECTED_ABNORMAL}）" if st['present'] else ""))
        print("  " + ("完整。" if st["ok"] else "不完整——重跑本命令继续下载。"))

    print("\n生成指纹清单 …")
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    entries = {}
    for _, p, _sz in sorted(jobs, key=lambda x: x[1]):
        if os.path.exists(p):
            rel = os.path.relpath(p, args.dest).replace(os.sep, "/")
            entries[rel] = {"sha256": sha256_of(p), "bytes": os.path.getsize(p)}
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump({"source": BASE, "dest": args.dest, "n_files": len(entries),
                   "files": entries}, fh, ensure_ascii=False, indent=2)
    print(f"  {len(entries)} 个文件的 SHA256 已写入 {args.manifest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。重跑同一命令即可从断点继续。")
        sys.exit(130)
