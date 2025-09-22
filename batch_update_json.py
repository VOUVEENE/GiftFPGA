#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 test/ 目录下批量修改 JSON 参数（按是否包含 _no_gift 后缀分两批）并校验结果。

功能要点：
- 支持点路径键（如 "use_gift_init_place" 或 "global_place_stages.0.iteration"）
- 两批文件可分别指定不同的新值（--value-gift / --value-no-gift）
- 支持 dry-run（只显示将要发生的变化）、仅查看当前值（不修改）
- 写入前自动可选备份，写入后自动重新读取并校验新值
- 可选严格模式（发现缺键则记为失败并返回非零退出码）
"""

import argparse, json, sys, time, pathlib, traceback

def parse_value(raw: str):
    """优先按 JSON 字面量解析（数值、true/false/null、数组、对象），失败则按字符串处理。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw

def get_by_path(obj, path_tokens):
    """根据点路径读取值（支持对象和列表）。"""
    cur = obj
    for tok in path_tokens:
        if isinstance(cur, list):
            try:
                idx = int(tok)
            except ValueError:
                raise KeyError(f"路径 '{tok}' 期望为列表索引")
            if idx < 0 or idx >= len(cur):
                raise KeyError(f"列表索引越界 '{tok}'")
            cur = cur[idx]
        else:
            if tok not in cur:
                raise KeyError(f"缺少键 '{tok}'")
            cur = cur[tok]
    return cur

def set_by_path(obj, path_tokens, new_value, create_missing=False):
    """根据点路径设置值（支持对象和列表），可选缺键时自动创建。"""
    cur = obj
    for i, tok in enumerate(path_tokens):
        is_last = (i == len(path_tokens) - 1)
        if is_last:
            if isinstance(cur, list):
                idx = int(tok)
                if idx < 0 or idx >= len(cur):
                    raise KeyError(f"列表索引越界 '{tok}'")
                cur[idx] = new_value
            else:
                cur[tok] = new_value
        else:
            if isinstance(cur, list):
                idx = int(tok)
                if idx < 0 or idx >= len(cur):
                    raise KeyError(f"列表索引越界 '{tok}'")
                nxt = cur[idx]
            else:
                if tok not in cur:
                    if create_missing:
                        nxt_tok = path_tokens[i+1]
                        # 下一个 token 可转为 int 则创建列表，否则创建对象
                        try:
                            int(nxt_tok)
                            cur[tok] = []
                        except ValueError:
                            cur[tok] = {}
                    else:
                        raise KeyError(f"缺少键 '{tok}'")
                nxt = cur[tok]
            cur = nxt

def process_files(files, path_tokens, new_value, opts, group_name):
    """处理一批文件：读 -> 改 -> 写 -> 校验，返回统计。"""
    examined = 0
    changed = 0
    failures = []
    ts = time.strftime("%Y%m%d-%H%M%S")

    for jf in files:
        examined += 1
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[{group_name}][FAIL] {jf}: 无法解析 JSON: {e}")
            failures.append((str(jf), "parse_error"))
            continue

        # 取当前值
        had_key = True
        current_val = None
        try:
            current_val = get_by_path(data, path_tokens)
        except Exception:
            had_key = False

        # 仅查看模式
        if opts.list_values:
            disp = json.dumps(current_val) if had_key else "<缺少该键>"
            print(f"[{group_name}][LIST] {jf}: {disp}")
            continue

        # 是否需要修改
        needs_change = (not had_key) or (current_val != new_value)
        if not needs_change:
            print(f"[{group_name}][SKIP] {jf}: 已是目标值 {json.dumps(new_value, ensure_ascii=False)}")
            continue

        # 尝试设置
        try:
            set_by_path(data, path_tokens, new_value, create_missing=opts.create_missing)
        except Exception as e:
            msg = f"设置失败: {e}"
            if opts.strict:
                print(f"[{group_name}][FAIL] {jf}: {msg}")
                failures.append((str(jf), "set_failed"))
            else:
                print(f"[{group_name}][WARN] {jf}: {msg}（已跳过）")
            continue

        # dry-run 不落盘
        if opts.dry_run:
            before = json.dumps(current_val, ensure_ascii=False) if had_key else "<缺少该键>"
            after = json.dumps(new_value, ensure_ascii=False)
            print(f"[{group_name}][DRY]  {jf}: {before} -> {after}")
            changed += 1
            continue

        # 备份
        if opts.backup:
            bak = jf.with_suffix(jf.suffix + f".bak.{ts}")
            try:
                bak.write_text(jf.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception as e:
                print(f"[{group_name}][WARN] {jf}: 备份失败: {e}")

        # 写回
        try:
            jf.write_text(json.dumps(data, ensure_ascii=False, indent=opts.indent) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[{group_name}][FAIL] {jf}: 写入失败: {e}")
            failures.append((str(jf), "write_failed"))
            continue

        # 校验
        try:
            verify = json.loads(jf.read_text(encoding="utf-8"))
            got = get_by_path(verify, path_tokens)
            if got != new_value:
                print(f"[{group_name}][FAIL] {jf}: 校验不一致（读到 {got!r}）")
                failures.append((str(jf), "verify_mismatch"))
                continue
        except Exception as e:
            print(f"[{group_name}][FAIL] {jf}: 校验失败: {e}")
            failures.append((str(jf), "verify_error"))
            continue

        before = json.dumps(current_val, ensure_ascii=False) if had_key else "<缺少该键>"
        after = json.dumps(new_value, ensure_ascii=False)
        print(f"[{group_name}][OK]   {jf}: {before} -> {after}")
        changed += 1

    return examined, changed, failures

def main():
    ap = argparse.ArgumentParser(description="按是否包含 _no_gift 后缀分两批修改 test 下 JSON 的同一参数。")
    ap.add_argument("--dir", default="test", help="JSON 文件目录（默认：test）")
    ap.add_argument("--key", required=True, help="点路径键，如 use_gift_init_place 或 global_place_stages.0.iteration")
    ap.add_argument("--value-gift", help="【有 GiFt 批】的新值（JSON 字面量优先；示例：1、true、0.01、\"str\"）")
    ap.add_argument("--value-no-gift", help="【no_gift 批】的新值（同上）")
    ap.add_argument("--dry-run", action="store_true", help="试运行，不写文件，只显示会发生的变化")
    ap.add_argument("--list-values", action="store_true", help="仅查看当前值，不做修改")
    ap.add_argument("--create-missing", action="store_true", help="若路径缺失则自动创建嵌套键/列表")
    ap.add_argument("--backup", action="store_true", help="写入前为每个文件生成 .bak 时间戳备份")
    ap.add_argument("--indent", type=int, default=4, help="写回 JSON 的缩进（默认 4）")
    ap.add_argument("--strict", action="store_true", help="严格模式：路径不存在等视为失败并返回非零码")
    args = ap.parse_args()

    base = pathlib.Path(args.dir)
    if not base.exists():
        print(f"[ERROR] 目录不存在：{base}")
        sys.exit(2)

    all_json = sorted([p for p in base.glob("*.json") if p.is_file()])
    if not all_json:
        print(f"[ERROR] 未在 {base}/ 下找到任何 .json 文件")
        sys.exit(2)

    # 按是否包含 _no_gift.json 划分两批
    files_no_gift = [p for p in all_json if p.name.endswith("_no_gift.json")]
    files_gift = [p for p in all_json if not p.name.endswith("_no_gift.json")]

    if args.list_values:
        # 仅查看时不强制要求提供新值
        value_gift = value_no_gift = None
    else:
        if args.value_gift is None or args.value_no_gift is None:
            print("[ERROR] 修改模式下需要同时提供 --value-gift 与 --value-no-gift")
            sys.exit(2)
        value_gift = parse_value(args.value_gift)
        value_no_gift = parse_value(args.value_no_gift)

    path_tokens = args.key.split(".")

    # 处理两批
    total_examined = 0
    total_changed = 0
    total_failures = []

    if files_no_gift:
        vg = value_no_gift
        if args.list_values:
            vg = None
        ex, ch, fail = process_files(files_no_gift, path_tokens, vg, args, "NO_GIFT")
        total_examined += ex; total_changed += ch; total_failures.extend(fail)
    else:
        print("[NO_GIFT] 未找到 *_no_gift.json 文件")

    if files_gift:
        vg = value_gift
        if args.list_values:
            vg = None
        ex, ch, fail = process_files(files_gift, path_tokens, vg, args, "GIFT")
        total_examined += ex; total_changed += ch; total_failures.extend(fail)
    else:
        print("[GIFT] 未找到非 *_no_gift.json 的文件")

    # 汇总
    if args.list_values:
        print(f"\n[SUMMARY] 共检查 {total_examined} 个文件；仅查看完成。解析/路径错误数：{len(total_failures)}")
        sys.exit(0 if not total_failures else 1)
    else:
        print(f"\n[SUMMARY] 共处理 {total_examined} 个文件，修改 {total_changed} 个，失败 {len(total_failures)} 个")
        sys.exit(0 if not total_failures else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中止] 用户取消")
        sys.exit(130)
    except Exception:
        print("[致命错误] 未预期异常：\n" + "".join(traceback.format_exc()))
        sys.exit(3)
