from __future__ import annotations

"""
微博热搜半衰期研究 — 数据采集脚本 v3
从 weibotop.cn API 抓取 2019 年 500 个热搜词条的完整热度曲线数据
v3 修复：断点续传复用 Phase1 结果 | 脏数据不入库 | 每天早中晚三次采样 | 数值化在榜时长
"""
import hashlib
import base64
import json
import csv
import random
import time
import asyncio
import os
from datetime import datetime, timedelta

import aiohttp
from tqdm import tqdm
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# ============================================================
# Section 1 — AES 加解密
# ============================================================

_KEY = bytes.fromhex(hashlib.sha1(
    "tSdGtmwh49BcR1irt18mxG41dGsBuGKS".encode()
).hexdigest()[:32])


def encrypt(plain: str | None) -> str | None:
    if plain is None:
        return None
    cipher = AES.new(_KEY, AES.MODE_ECB)
    return base64.b64encode(cipher.encrypt(pad(plain.encode(), AES.block_size))).decode()


def decrypt(b64: str | None):
    if not b64 or b64.strip() == "":
        return None
    cipher = AES.new(_KEY, AES.MODE_ECB)
    raw = base64.b64decode(b64)
    return json.loads(unpad(cipher.decrypt(raw), AES.block_size).decode())


# ============================================================
# Section 2 — 异步 API 客户端（重试 + 退避）
# ============================================================

BASE = "https://api.weibotop.cn"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.weibotop.cn/2.0/",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint_2019.json")

RANDOM_SEED = 42  


async def api_get(session: aiohttp.ClientSession, path: str, **params) -> str:
    """
    带指数退避的 GET 请求。
    自动检测 502 / Invalid / 空响应并重试，最多 8 次（最长等待 ~5 分钟）。
    """
    url = f"{BASE}/{path}"
    last_err = ""
    for attempt in range(8):
        try:
            async with session.get(
                url, params=params, headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                text = await resp.text()

            if resp.status == 502 or "502 Bad Gateway" in text:
                last_err = "502"
                wait = min(2 ** attempt + random.uniform(0, 1), 60)
                if attempt == 0:
                    print(f"  [!] API 502，等待恢复...", flush=True)
                await asyncio.sleep(wait)
                continue

            if text.strip().startswith("Invalid"):
                last_err = f"DCE: {text[:50]}"
                await asyncio.sleep(min(2 ** attempt + random.uniform(0, 1), 30))
                continue

            if not text or text.strip() == "":
                last_err = "empty"
                await asyncio.sleep(2 ** attempt)
                continue

            return text

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = str(e)[:80]
            await asyncio.sleep(min(2 ** attempt + random.uniform(0.5, 1.5), 30))

    raise RuntimeError(f"API {path} failed after 8 retries: {last_err}")


async def get_timeid_for_timestamp(session, ts: str) -> tuple[str, str]:
    """ts='2019-11-15 08:00:00' → (timeid, actual_timestamp)"""
    text = await api_get(session, "getclosesttime", timestamp=encrypt(ts))
    data = json.loads(text)
    return str(data[0]), str(data[1])


async def get_topics(session, timeid: str) -> list[dict]:
    text = await api_get(session, "currentitems", timeid=encrypt(timeid))
    rows = decrypt(text)
    if rows is None:
        return []
    topics = []
    for r in rows:
        try:
            name = str(r[0]).strip()
            downtime = str(r[1]).replace(".0", "").strip()
            uptime = str(r[2]).replace(".0", "").strip()
            hotindex = int(r[3])
            if name and uptime and downtime:
                topics.append({
                    "name": name, "uptime": uptime,
                    "downtime": downtime, "hotindex": hotindex,
                })
        except (IndexError, ValueError):
            continue
    return topics


async def get_rank_history(session, name: str):
    text = await api_get(session, "getrankhistory", name=encrypt(name))
    data = decrypt(text)
    if data and len(data) >= 3 and data[0]:
        return data[0], data[1], data[2]
    return None


# ============================================================
# Section 3 — 工具函数
# ============================================================

def calc_duration(uptime: str, downtime: str) -> tuple[str, int]:
    """返回 (文本格式, 秒数)"""
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        dt = datetime.strptime(downtime, fmt) - datetime.strptime(uptime, fmt)
        total_sec = int(dt.total_seconds())
        h, m, s = total_sec // 3600, (total_sec % 3600) // 60, total_sec % 60
        return f"{h}时{m}分{s}秒", total_sec
    except Exception:
        return "", 0


def calc_half_life(timestamps: list[str], heats: list[str]) -> float | None:
    """
    计算热度半衰期（秒）：从峰值首次衰减到 50% 所经过的时间。

    算法说明：
    1. 在热度序列中找到全局最大值作为"峰值"
    2. 若存在多个相等的全局最大值，取首次出现的那个作为衰减起点
       （第一个达到峰值的时刻 = 热度开始衰退的转折点）
    3. 从该峰值时间点向后扫描，找到第一个 ≤ 峰值 50% 的时间点
    4. 返回两个时间点的差值（秒）；若热度从未跌至 50% 以下，返回 None
    """
    if not timestamps or len(timestamps) < 2:
        return None

    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        vals = [float(h) for h in heats]
        peak = max(vals)
        # 以首次达到绝对最高热度的时间点作为半衰期计算基准
        # （双峰等高时取第一个峰，因为那是热度衰退的真正起点）
        peak_idx = vals.index(peak)
        threshold = peak * 0.5

        for i in range(peak_idx + 1, len(vals)):
            if vals[i] <= threshold:
                t_peak = datetime.strptime(timestamps[peak_idx].replace(".0", ""), fmt)
                t_half = datetime.strptime(timestamps[i].replace(".0", ""), fmt)
                return (t_half - t_peak).total_seconds()

        return None
    except Exception:
        return None


# ============================================================
# Section 4 — 断点续传
# ============================================================

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"topics_basic": [], "topics_with_curve": [], "failed_curves": []}


def save_checkpoint(state: dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# Section 5 — Phase 1: 多时段采样 + 基础词条库构建
# ============================================================

# 每天分早、中、晚三个时段各请求一次 timeid，
# 以覆盖生命周期短（只活跃几小时）的热搜词条
TIME_SLOTS = ["08:00:00", "14:00:00", "20:00:00"]


async def phase1_collect_topics(session, num_dates: int = 25) -> list[dict]:
    ck = load_checkpoint()
    if ck.get("topics_basic"):
        print(f"[Phase 1] 复用断点中的基础词条库 ({len(ck['topics_basic'])} 条，跳过 API)")
        return ck["topics_basic"]

    start = datetime(2019, 10, 25)
    end = datetime(2019, 12, 31)
    all_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
                 for i in range((end - start).days + 1)]
    sampled = random.sample(all_dates, min(num_dates, len(all_dates)))
    print(f"[Phase 1] 从 {len(all_dates)} 个日期中随机选 {len(sampled)} 个 "
          f"× {len(TIME_SLOTS)} 时段 = {len(sampled) * len(TIME_SLOTS)} 次请求")

    seen_names = set()
    all_topics: list[dict] = []

    pbar = tqdm(sampled, desc="  Phase 1", unit="day", ncols=80)
    for date_str in pbar:
        day_topics: list[dict] = []
        for slot in TIME_SLOTS:
            try:
                ts = f"{date_str} {slot}"
                timeid, _ = await get_timeid_for_timestamp(session, ts)
                topics = await get_topics(session, timeid)
                day_topics.extend(topics)
            except Exception:
                pass  # 个别时段失败不中断，继续抓其他时段
            await asyncio.sleep(0.2)

        # 当天内去重
        day_seen = set()
        day_unique: list[dict] = []
        for t in day_topics:
            if t["name"] not in day_seen:
                day_seen.add(t["name"])
                day_unique.append(t)

        new_count = 0
        for t in day_unique:
            if t["name"] not in seen_names:
                seen_names.add(t["name"])
                all_topics.append(t)
                new_count += 1

        pbar.set_postfix_str(f"累计 {len(all_topics)} 条")

    # 保存进断点
    ck["topics_basic"] = all_topics
    save_checkpoint(ck)
    print(f"  → 合计去重词条: {len(all_topics)}（已存入断点）", flush=True)
    return all_topics


# ============================================================
# Section 6 — Phase 2: 热度曲线抓取
# ============================================================

async def phase2_fetch_curves(session, topics: list[dict],
                               target: int = 500) -> list[dict]:
    sample = random.sample(topics, min(target, len(topics)))
    print(f"[Phase 2] 获取 {len(sample)} 个词条的热度曲线...", flush=True)

    ck = load_checkpoint()
    done_names = {t["name"] for t in ck["topics_with_curve"]}
    done_names.update(ck.get("failed_curves", []))

    results: list[dict] = list(ck["topics_with_curve"])
    failed: list[str] = list(ck.get("failed_curves", []))

    pending = [t for t in sample if t["name"] not in done_names]
    if not pending:
        print(f"  → 已有 {len(results)} 条缓存，全部完成", flush=True)
        return results

    print(f"  → 已有 {len(results)} 条缓存, 待抓取 {len(pending)} 条", flush=True)

    pbar = tqdm(pending, desc="  Phase 2", unit="条", ncols=80)
    for idx, topic in enumerate(pbar):
        try:
            raw = await get_rank_history(session, topic["name"])
            if raw:
                topic["timestamps"], topic["ranks"], topic["heats"] = raw
                results.append(topic)
            else:
                failed.append(topic["name"])
        except Exception:
            failed.append(topic["name"])

        if (idx + 1) % 20 == 0:
            save_checkpoint({
                "topics_basic": topics,
                "topics_with_curve": results,
                "failed_curves": failed,
            })

        pbar.set_postfix_str(f"✓{len(results)} ✗{len(failed)}")
        await asyncio.sleep(0.3)

    # 最终保存
    save_checkpoint({
        "topics_basic": topics,
        "topics_with_curve": results,
        "failed_curves": failed,
    })
    print(f"  → 完成: {len(results)} 条有效, {len(failed)} 条失败", flush=True)
    return results


# ============================================================
# Section 7 — Phase 3: CSV 输出
# ============================================================

def phase3_output(topics: list[dict], out_path: str):
    print(f"[Phase 3] 输出 CSV ...", flush=True)
    valid_topics = [t for t in topics if t.get("timestamps")]
    max_len = max((len(t["timestamps"]) for t in valid_topics), default=0)
    print(f"  → 有效词条: {len(valid_topics)}/{len(topics)}, 最大曲线点数: {max_len}", flush=True)

    curve_cols = []
    for i in range(max_len):
        curve_cols.append(f"t_{i+1}")
        curve_cols.append(f"heat_{i+1}")

    headers = [
        "词条名", "上榜时间", "最后在榜时间",
        "在榜共计", "在榜共计_秒",      # 文本 + 纯数字双列
        "热度最大值", "半衰期_秒",
        *curve_cols,
    ]

    rows = []
    for t in topics:
        ts = t.get("timestamps", [])
        hs = t.get("heats", [])
        duration_text, duration_sec = calc_duration(
            t.get("uptime", ""), t.get("downtime", "")
        )
        max_heat = max((float(h) for h in hs), default=None)
        half_life = calc_half_life(ts, hs)

        row = {
            "词条名": t["name"],
            "上榜时间": t.get("uptime", ""),
            "最后在榜时间": t.get("downtime", ""),
            "在榜共计": duration_text,
            "在榜共计_秒": duration_sec,
            "热度最大值": max_heat,
            "半衰期_秒": half_life if half_life is not None else "",
        }
        for i in range(max_len):
            row[f"t_{i+1}"] = ts[i].replace(".0", "") if i < len(ts) else ""
            row[f"heat_{i+1}"] = hs[i] if i < len(hs) else ""
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → 输出: {out_path}", flush=True)
    valid_hl = [r for r in rows if r["半衰期_秒"] != ""]
    if valid_hl:
        avg = sum(float(r["半衰期_秒"]) for r in valid_hl) / len(valid_hl)
        print(f"  → 有效半衰期: {len(valid_hl)}/{len(rows)}, "
              f"均值: {avg:.0f}秒 ({avg/3600:.1f}小时)", flush=True)


# ============================================================
# Section 8 — Main
# ============================================================

async def main():
    random.seed(RANDOM_SEED)
    t0 = time.time()
    print("=" * 60, flush=True)
    print("微博热搜半衰期研究 — 数据采集 v3", flush=True)
    print(f"目标: 2019-10-25 ~ 2019-12-31, 500 个词条 (seed={RANDOM_SEED})", flush=True)
    print("=" * 60, flush=True)

    connector = aiohttp.TCPConnector(limit=5, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:

        # Phase 1：断点有缓存则跳过 API
        all_topics = await phase1_collect_topics(session, num_dates=25)
        print()

        # Phase 2：抓取热度曲线（断点续传）
        results = await phase2_fetch_curves(session, all_topics, target=500)
        print()

    # Phase 3：输出 CSV
    out_path = os.path.join(BASE_DIR, "weibo_hotsearch_2019.csv")
    phase3_output(results, out_path)

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f} 秒 ({elapsed/60:.1f} 分钟)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
