import re
import os
import json
import traceback
import time
import uuid
import argparse
from collections import defaultdict, deque
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if elapsed > 0.1:
            print(f"[PERF] {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

# ---- 把原来的 USE_RE / DEF_RE 换成这两个更稳的版本 ----
USE_RE = re.compile(r'^\[USE\]\s+(\d+)\s*\|\s*@([$\w@]+)\s*\|\s*current=<([^>]+)>\s*(?:\|\s*target=([^|]*)\s*)?\|\s*Type=(\d+)')

DEF_RE = re.compile(r'^\[DEF\]\s+(\d+)\s*\|\s*@([$\w@]+).*origin=<([^>]+)>\s*\|\s*Type=(\d+)')

# ---- 备用的“宽松”匹配（没有 Type= 也能过）----
USE_RE_LOOSE = re.compile(
    r'^\[USE\]\s+(\d+)\s*'
    r'\|\s*@([^|]+?)\s*'
    r'\|\s*current\s*=\s*(?:<([^>]+)>|([^|]+?))\s*'
    r'(?:\|\s*target\s*=\s*(?:<([^>]+)>|([^|]+?))\s*)?\s*$'
)

DEF_RE_LOOSE = re.compile(
    r'^\[DEF\]\s+(\d+)\s*'
    r'\|\s*@([^|]+?).*?'
    r'\|\s*origin\s*=\s*(?:<([^>]+)>|([^|]+?))\s*\s*$'
)


ENTRY_RE = re.compile(
    r'^\[Entry\]\s+(\d+)\s*\|\s*(.+)$'
)

EXIT_RE = re.compile(
    r'^\[Exit\]\s+(\d+)\s*\|\s*(.+)$'
)

CALL_RE = re.compile(
    r'^\[CALL\]\s+(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+)\s*$'
)

CFG_RE = re.compile(
    r'^\[CFG\]\s*(\d+)\s*(?:→|->)\s*(\d+)\s*\|\s*(?:<([^>]+)>|(.+))\s*$'
)

class DependencyNode:
    __slots__ = (
        "id","line","filename","method","defs","uses","calls","cfg",
        "is_entry","is_exit","timestamp","in_edges"
    )
    def __init__(self, node_id, line, filename, method, timestamp):
        self.id = node_id
        self.line = line
        self.filename = filename
        self.method = method
        self.defs = defaultdict(lambda: {'count': 0, 'type': None, 'origin': None})
        self.uses = defaultdict(lambda: {'count': 0, 'type': 0})
        self.calls = defaultdict(int)
        self.cfg = defaultdict(int)
        self.is_entry = False
        self.is_exit = False
        self.timestamp = timestamp
        self.in_edges = set()

    def add_in_edge(self, src_line):
        self.in_edges.add(src_line)

class DependencyAnalyzer:
    def __init__(self, debug=False):
        self.graphs = defaultdict(lambda: {
            'nodes': defaultdict(list),
            'edges': defaultdict(lambda: defaultdict(int))
        })
        self.def_registry = defaultdict(lambda: {
            'entries': deque(maxlen=100),
            'local_vars': defaultdict(lambda: deque(maxlen=3000)),
            'class_vars': {}
        })
        self.call_stack = []  # (调用文件, 调用行, 返回行)
        self.method_entries = defaultdict(dict)  # filename -> {methodSig: entryLine}
        self.current_method = None
        self.line_counter = 0
        self.debug = debug
        self.call_args = defaultdict(lambda: defaultdict(list))
        self.pending_exit = None
        self.line_vars = defaultdict(set)
        self.def_cache = defaultdict(lambda: {
            'class_vars': {},
            'local_vars': defaultdict(deque)
        })
        self.success_count = 0
        self.failure_count = 0
        self.call_exit = []
        self.number = 0
        self._filename_cache = {}

        self.file_cfg_nodes = defaultdict(set)
        self.file_cfg_edges = defaultdict(lambda: defaultdict(int))

        self.recent_uses = deque(maxlen=50)
        self.current_timestamp = 0
        self.edge_history = defaultdict(lambda: defaultdict(int))
        self.node_id_counter = 1000  # 初始值设为1000以便区分

        # ==== 新增：方法上下文与行级方法 hint（不改变你的造点/边逻辑）====
        self._method_ctx = defaultdict(list)          # filename -> [methodSig,...] 栈
        self._line_method_hint = defaultdict(dict)    # filename -> { line:int -> methodSig }
        # --- CFG 组压缩：同一目标的连续CFG先缓冲，结束时只保留1条 ---
        self._cfg_group = None   # 当前正在缓冲的 {filename, method, dst, candidates=[(src_line, seq), ...]}
        self._cfg_seq = 0        # 递增序号，模拟“出现次序”，用于最近性判定
        # 维护“最近一次有CFG指向该行”的出现序号：key = (filename, line)
        self._last_incoming = defaultdict(lambda: -1)
        self._call_key_counts = defaultdict(int)  # key=(callee_method, callee_file) -> 当前栈里此类帧的数量
        self._warn_seen = defaultdict(int)        # 节流同类WARN

    # ---------------- 工具：方法上下文 & 行级 hint ----------------
    def _push_method(self, filename, method_sig):
        if method_sig:
            self._method_ctx[filename].append(method_sig)

    def _pop_method(self, filename):
        stk = self._method_ctx.get(filename)
        if stk:
            stk.pop()

    def _peek_method(self, filename):
        stk = self._method_ctx.get(filename)
        return stk[-1] if stk else None

    def _record_method_hint(self, filename, line_num, method_sig):
        # 只记录非空
        if method_sig:
            self._line_method_hint[filename][line_num] = method_sig

    def _hint_for(self, filename, line_num, fallback=None):
        # 优先行级 hint，其次当前方法栈，最后外部给的 fallback
        return (self._line_method_hint[filename].get(line_num)
                or self._peek_method(filename)
                or fallback
                or "unknown")

    def _link_existing_nodes(self, filename, src_node, dst_node, edge_type, var=None):
        """只在给定的两个已存在节点之间连边；不会创建新节点，也不会触发分裂逻辑。"""
        timestamp = self._get_new_timestamp()
        dst_node.timestamp = timestamp
        src_node.timestamp = timestamp
        dst_node.add_in_edge(src_node.id)

        edge_key = (edge_type, var or '')
        self.graphs[filename]['edges'][(src_node.id, dst_node.id)][edge_key] += 1
        self.edge_history[filename][(src_node.line, dst_node.line)] = timestamp

    # ---------------- 日志/文件名工具 ----------------
    def log(self, message, level="INFO"):
        if self.debug or level in ("ERROR", "WARN"):
            print(f"[{level}] {message}")

    def sanitize_filename(self, filename):
        sanitized = re.sub(r'[^a-zA-Z0-9/_.-]', '_', filename).replace('<', '_').replace('>', '_')
        return sanitized

    def get_filename(self, method_signature):
        """
        将任意方法/类签名映射到该签名所属“顶层类”的源文件路径：
        e.g.  a.b.C$Inner -> a/b/C.java
            a.b.C$$Lambda$23/0x... -> a/b/C.java
            <a.b.C: void m()> -> a/b/C.java
        """
        try:
            if not method_signature:
                return "unknown.java"

            cache = self._filename_cache
            ms = method_signature.strip()
            if ms in cache:
                return cache[ms]

            s = ms
            # 去掉尖括号包裹（如 <a.b.C: void m()>）
            if s.startswith('<') and s.endswith('>'):
                s = s[1:-1]

            # 取“类全名”部分（去掉方法签名）
            if '#' in s:
                class_part = s.split('#', 1)[0]
            elif ':' in s:
                class_part = s.split(':', 1)[0]
            else:
                # 可能是单独的类名，或前后有多余空白
                class_part = s.split()[0]

            # 统一分隔符，防止偶发的斜杠写法
            class_part = class_part.replace('/', '.').strip()

            # 关键：映射到“顶层类”（丢弃第一个 '$' 及其后缀）
            # 适配内部类、匿名类、CGLIB、lambda 等命名（C$$Lambda$...）
            if '$' in class_part:
                class_part = class_part.split('$', 1)[0]

            filename = class_part.replace('.', '/') + '.java'
            filename = self.sanitize_filename(filename)

            cache[ms] = filename
            return filename
        except Exception:
            self.log(f"Error parsing method signature: {method_signature}", "ERROR")
            return "unknown.java"



    def _pop_matching_call_frame(self, callee_method, callee_file):
        """从栈顶向下找到与当前 Exit 匹配的调用帧（方法签名+文件都匹配），并弹出它。"""
        for i in range(len(self.call_stack) - 1, -1, -1):
            fr = self.call_stack[i]
            if (fr.get('callee_method') == callee_method
                    and fr.get('callee_file') == callee_file):
                return self.call_stack.pop(i)
        return None  # 没找到就返回 None
    # ---------------- 解析 ----------------
    def parse_line(self, line):
        self.line_counter += 1
        try:
            if not line or line[0] != '[':
                self.failure_count += 1
                return None

            # 先用 startswith 决定“只尝试一个正则”
            if line.startswith('[USE]'):
                m = USE_RE.match(line) or USE_RE_LOOSE.match(line)
                if not m:
                    self.failure_count += 1
                    return None
                current = (m.group(3) or m.group(4) or '').strip()
                target  = (m.group(5) or m.group(6) or '').strip() or None
                use_type = int(m.group(7)) if m.lastindex and m.lastindex >= 7 and m.group(7) is not None else 0
                return {'type':'USE','line':int(m.group(1)),'var':m.group(2),
                        'current_method': current,'target': target,'use_type': use_type}

            if line.startswith('[DEF]'):
                m = DEF_RE.match(line) or DEF_RE_LOOSE.match(line)
                if not m:
                    self.failure_count += 1
                    return None
                origin = (m.group(3) or m.group(4) or '').strip()
                def_type = int(m.group(5)) if m.lastindex and m.lastindex >= 5 and m.group(5) is not None else 0
                return {'type':'DEF','line':int(m.group(1)),'var':m.group(2),
                        'origin':origin,'def_type':def_type}

            if line.startswith('[Entry]'):
                m = ENTRY_RE.match(line)
                if not m:
                    self.failure_count += 1
                    return None
                return {'type':'ENTRY','line':int(m.group(1)),'method':m.group(2).strip()}

            if line.startswith('[Exit]'):
                m = EXIT_RE.match(line)
                if not m:
                    self.failure_count += 1
                    return None
                return {'type':'EXIT','line':int(m.group(1)),'method':m.group(2).strip()}

            if line.startswith('[CALL]'):
                m = CALL_RE.match(line)
                if not m:
                    self.failure_count += 1
                    return None
                return {'type':'CALL','line':int(m.group(1)),
                        'caller':m.group(2).strip(),'callee':m.group(3).strip()}

            if line.startswith('[CFG]'):
                m = CFG_RE.match(line)
                if not m:
                    self.failure_count += 1
                    return None
                method = (m.group(3) or m.group(4) or '').strip()
                return {'type':'CFG','line':int(m.group(1)),
                        'target':int(m.group(2)),'method':method}

            self.failure_count += 1
            return None
        except Exception:
            self.failure_count += 1
            self.log(f"解析异常 (L{self.line_counter})", "ERROR")
            return None


    @timeit
    def process_logs(self, log_path):
        self.log(f"开始处理日志文件: {log_path}")
        try:
            # 16MB 缓冲；把热函数绑定到局部变量
            process_line = self._process_line
            with open(log_path, 'r', encoding='utf-8', errors='ignore', buffering=16*1024*1024) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line_num % 1000 == 0:
                            self.log(f"处理行 {line_num}")
                        process_line(line.strip())
                        self.success_count += 1
                    except Exception:
                        self.failure_count += 1
                        self.log(f"处理行 {line_num} 失败", "ERROR")
        except FileNotFoundError:
            self.log(f"文件未找到: {log_path}", "ERROR")
        except Exception as e:
            self.log(f"处理日志时发生严重错误: {str(e)}", "ERROR")

        if getattr(self, '_cfg_group', None):
            self._flush_cfg_group()
        return self.graphs


    def _process_line(self, line):
        if not line:
            return
        record = self.parse_line(line)
        if not record:
            return
        try:
            if record['type'] != 'CFG' and getattr(self, '_cfg_group', None):
                self._flush_cfg_group()

            handler = getattr(self, f"handle_{record['type'].lower()}", None)
            if handler:
                handler(record)
            else:
                self.failure_count += 1
                self.log(f"未知记录类型: {record['type']}", "WARN")
        except Exception as e:
            self.failure_count += 1
            self.log(f"处理记录失败: {traceback.format_exc()}", "ERROR")

    # ---------------- 时间戳/节点工具（不改变你的造点策略） ----------------
    def _get_new_timestamp(self):
        self.current_timestamp += 1
        return self.current_timestamp

    def _generate_node_id(self):
        self.node_id_counter += 1
        return f"N{self.node_id_counter}"

    def _get_latest_node(self, filename, line_num, method_hint=None):
        # 若未显式提供，使用行级 hint 或方法栈
        if not method_hint:
            method_hint = self._hint_for(filename, line_num)
        nodes = self.graphs[filename]['nodes'][line_num]
        if not nodes:
            return self._create_node(filename, line_num, method_hint or "unknown")
        node = max(nodes, key=lambda x: x.timestamp)
        # 给该行所有 unknown 节点补标（不影响结构）
        if method_hint:
            for n in nodes:
                if not n.method or n.method == "unknown":
                    n.method = method_hint
        return node

    def _create_node(self, filename, line_num, method):
        # 关键：创建时就尽力确定方法名（不改变你的额外造点策略）
        if not method or method == "unknown":
            method = self._hint_for(filename, line_num, method)
        new_id = self._generate_node_id()
        new_node = DependencyNode(new_id, line_num, filename, method, self._get_new_timestamp())
        self.graphs[filename]['nodes'][line_num].append(new_node)
        return new_node

    # ---------------- 加边（保持你的原逻辑，不改分支/过滤） ----------------
    def _add_edge(self, src_file, src_line, dst_line, edge_type, var=None):
        """保持你的原实现，只在取节点时利用 hint 补方法名。"""
        timestamp = self._get_new_timestamp()

        # 源节点：用行级/方法栈 hint
        src_node = self._get_latest_node(src_file, src_line)

        # 保留你原来的历史连接/额外造点逻辑 ↓↓↓
        if timestamp - src_node.timestamp > 30 or (src_line > dst_line and edge_type=='cfg'):
            src_nodes = self.graphs[src_file]['nodes'][src_line]
            dst_nodes = self.graphs[src_file]['nodes'][dst_line]
            dst_node = None
            found=False

            for s_node in reversed(src_nodes):
                for d_node in reversed(dst_nodes):
                    if (s_node.id, d_node.id) in self.graphs[src_file]['edges']:
                        if  found == False or s_node.timestamp > src_node.timestamp:
                            src_node = s_node
                            dst_node = d_node
                            found=True

            if src_line > dst_line and not found:
                found = True
                dst_node = self._get_latest_node(src_file, dst_line)

            if not found:
                # 这里创建新节点：我们仅补方法名，不改造点策略
                dst_node = self._create_node(src_file, dst_line, self._hint_for(src_file, dst_line))
        else:
            if edge_type =='data_flow' or edge_type=='class_use':
                dst_node=self._get_latest_node(src_file, dst_line)
            else:
                dst_nodes = self.graphs[src_file]['nodes'].get(dst_line, [])
                dst_node = None
                for node in reversed(dst_nodes):
                    # —— 保留你的原过滤（即便它会促使“额外造点”）——
                    non_entry_edges = [e for e in node.in_edges if e[1] != 'data_flow' or e[1] != 'class_use']
                    if len(non_entry_edges) == 0 or src_node.id in node.in_edges:
                        dst_node = node
                        break
            if not dst_node:
                dst_node = self._create_node(src_file, dst_line, self._hint_for(src_file, dst_line))

        # 更新时间戳/记录入边（保持你的原结构：只存 src_id）
        dst_node.timestamp = timestamp
        src_node.timestamp = timestamp
        dst_node.add_in_edge(src_node.id)

        edge_key = (edge_type, var or '')
        self.graphs[src_file]['edges'][(src_node.id, dst_node.id)][edge_key] += 1
        self.edge_history[src_file][(src_line, dst_line)] = timestamp

    def _add_both_edges(self, src_file, src_line, dst_file, dst_line, edge_type):
        if src_file != dst_file:
            return
        self._add_edge(src_file, src_line, dst_line, edge_type)

    # ---------------- 事件处理（只加 hint 记录，不改结构） ----------------
    def handle_entry(self, record):
        try:
            method_name = record['method']
            filename = self.get_filename(method_name)
            line_num = record['line']

            # 记录上下文 + 行级 hint（创建节点前）
            self._push_method(filename, method_name)
            self._record_method_hint(filename, line_num, method_name)
            self._record_method_hint(filename, line_num + 1, method_name)

            self.method_entries[filename][method_name] = line_num

            self._add_both_edges(filename, line_num, filename, line_num+1, 'entry')
            node = self._get_latest_node(filename, line_num, method_hint=method_name)
            node.is_entry = True
            self.current_method = method_name
        except Exception as e:
            self.log(f"处理Entry失败: {str(e)}", "ERROR")


    def _warn_miss_exit(self, key, exit_line):
        cnt = self._warn_seen[key]
        # 前 5 次都打，之后每 100 次打一条（你可以把100改成50/200）
        if cnt < 5 or cnt % 100 == 0:
            self.log(f"未找到匹配的CALL帧: Exit {key[0]} @{key[1]}:{exit_line}", "WARN")
        self._warn_seen[key] = cnt + 1
        
    def handle_exit(self, record):
        try:
            callee_method = record['method']   # 例如 <a.b.C: void m(int)>
            callee_file   = self.get_filename(callee_method)
            exit_line     = record['line']
            key = (callee_method, callee_file)

            # —— 若此方法/文件的调用帧计数为0，直接判定“必定无匹配”，避免 O(N) 扫栈 —— 
            if self._call_key_counts.get(key, 0) == 0:
                self._warn_miss_exit(key, exit_line)   # 节流打印，见下
                self._pop_method(callee_file)
                return
            # 补 hint（保证 Exit 行造点时能带上方法名）
            self._record_method_hint(callee_file, exit_line, callee_method)

            # 精确匹配这次 Exit 对应的调用帧
            frame = self._pop_matching_call_frame(callee_method, callee_file)
            if frame:
                # 仅当“同文件”时才连 return（与你现有 call/return 同文件策略保持一致）
                self._call_key_counts[key] -= 1
                if callee_file == frame['caller_file']:
                    # 源：callee 的 Exit 节点
                    src_node = self._get_latest_node(callee_file, exit_line, method_hint=callee_method)

                    # 目的：调用点“已存在”的那个节点（用 ID 精确定位），若不存在则尽量不造新点
                    dst_candidates = self.graphs[callee_file]['nodes'].get(frame['caller_line'], [])
                    dst_node = None
                    if dst_candidates:
                        # 先按 ID 精确找调用时那个节点
                        for n in dst_candidates:
                            if n.id == frame.get('caller_node_id'):
                                dst_node = n
                                break
                        # 如果因为其他原因找不到，退而选“已有的最后一个”（仍不分裂）
                        if dst_node is None:
                            dst_node = dst_candidates[-1]
                    else:
                        # 该行竟然没有节点（少见）：此时只能创建一个（不可避免）
                        dst_node = self._create_node(callee_file, frame['caller_line'],
                                                    self._hint_for(callee_file, frame['caller_line']))

                    # ★★ 关键：不走 _add_both_edges / _add_edge，直接在“已存在”节点之间连 return
                    self._link_existing_nodes(callee_file, src_node, dst_node, 'return')

                    # 正确地把 is_exit 标在“callee 的 Exit 节点”上
                    src_node.is_exit = True
                else:
                    self._warn_miss_exit(key, exit_line)
                    # 跨文件：按你当前策略跳过
                    pass
            else:
                self.log(f"未找到匹配的CALL帧: Exit {callee_method} @{callee_file}:{exit_line}", "WARN")

            # 弹出方法上下文
            self._pop_method(callee_file)
        except Exception as e:
            self.log(f"处理Exit失败: {str(e)}", "ERROR")



    def handle_call(self, record):
        try:
            caller = record['caller']
            callee = record['callee']
            caller_file = self.get_filename(caller)
            callee_file = self.get_filename(callee)
            call_line = record['line']

            # 在调用行打 hint（避免该行 first node 为 unknown）
            self._record_method_hint(caller_file, call_line, caller)

            # 先取得“调用点节点”（可能新建），并记录其 ID
            caller_node = self._get_latest_node(caller_file, call_line, method_hint=caller)
            caller_node.calls[callee] += 1
            
            key = (callee, callee_file)
            self._call_key_counts[key] += 1

            # 记录返回点栈帧（包含 callee 信息 + 调用点节点 ID，便于 Exit 精确匹配且不分裂）
            self.call_stack.append({
                'caller_file': caller_file,
                'caller_line': call_line,
                'caller_node_id': caller_node.id,  # ★ 关键：保存调用点节点 ID
                'callee_method': callee,
                'callee_file': callee_file,
            })

            # 若已知 callee 的 Entry 行，补一条 call 边（同文件才会真正落边）
            if callee in self.method_entries[callee_file]:
                entry_line = self.method_entries[callee_file][callee]
                self._add_both_edges(caller_file, call_line, callee_file, entry_line, 'call')

        except Exception as e:
            self.log(f"处理Call失败: {str(e)}", "ERROR")



    def handle_cfg(self, record):
        try:
            src_line = record['line']
            dst_line = record['target']
            method = record['method']
            if src_line == dst_line:
                return

            filename = self.get_filename(method)

            # 行级 hint：src/dst 都标记（确保新造点有方法名）
            self._record_method_hint(filename, src_line, method)
            self._record_method_hint(filename, dst_line, method)

            self._buffer_cfg(filename, src_line, dst_line, method)

            # if src_line != dst_line:
            #     self._add_both_edges(filename, src_line, filename, dst_line, 'cfg')

            # src_node = self._get_latest_node(filename, src_line, method_hint=method)
            # dst_node = self._get_latest_node(filename, dst_line, method_hint=method)
            # src_node.cfg[dst_line] += 1

        except Exception as e:
            self.log(f"处理CFG失败: {str(e)}", "ERROR")

    def _buffer_cfg(self, filename, src_line, dst_line, method):
        """把同一个 (filename, method, dst_line) 的连续CFG收集为一组。"""
        self._cfg_seq += 1
        seq = self._cfg_seq

        g = self._cfg_group
        # 同一文件、同一方法、同一目标行 => 继续收集
        if g and g['filename'] == filename and g['method'] == method and g['dst'] == dst_line:
            g['candidates'].append((src_line, seq))
        else:
            # 目标/方法/文件变化 => 先结算上一组，再开新组
            if g:
                self._flush_cfg_group()
            self._cfg_group = {
                'filename': filename,
                'method': method,
                'dst': dst_line,
                'candidates': [(src_line, seq)],
            }

    def _flush_cfg_group(self):
        """把缓冲的一组同目标CFG结算为唯一的一条边 F->A。"""
        g = self._cfg_group
        if not g:
            return

        filename = g['filename']
        method   = g['method']
        dst      = g['dst']
        candidates = g['candidates']   # [(src_line, seq), ...]

        # 1) 根据“向上最近一次有 X->F”的时间序号，从候选源 {B,C,D,...} 中挑 F
        best_src = None
        best_seen = -1
        for src, _ in candidates:
            seen = self._last_incoming[(filename, src)]
            if seen > best_seen:
                best_seen = seen
                best_src = src

        # 2) 若历史里没有任何 *->F，做一个稳妥的回退策略（取这组里“最后一个”候选）
        if best_src is None:
            best_src = candidates[-1][0]

        # 3) 仅添加唯一的一条 cfg 边：F -> A
        self._add_both_edges(filename, best_src, filename, dst, 'cfg')

        # 4) 维护“最近一次有人指向 A”的序号（用于后续‘向上最近’判断）
        #    用当前的全局序号（或递增计数）即可
        self._last_incoming[(filename, dst)] = self._cfg_seq

        # 5) 兼顾你原来记次数的逻辑（把真正的 src_node.cfg[dst] += 1）
        src_node = self._get_latest_node(filename, best_src, method_hint=method)
        src_node.cfg[dst] += 1

        # 6) 清空这一组
        self._cfg_group = None



    def handle_def(self, record):
        try:
            origin_method = record['origin']
            def_line = record['line']
            var_id = record['var']
            def_type = record['def_type']
            filename = self.get_filename(origin_method)

            # 行级 hint：定义行
            self._record_method_hint(filename, def_line, origin_method)

            if def_type == 0:
                self._record_local_def(filename, def_line, var_id, origin_method, 0, def_line)
                return

            true_def_line = def_line

            matched_use = self._find_previous_use(
                filename=filename,
                current_line=def_line,
                target_type=def_type,
                max_lines=20
            )

            if matched_use:
                use_var = matched_use['var']
                use_line = matched_use['line']
                guess_def = self._find_local_def(var_name=use_var, filename=filename, max_line=use_line)
                if guess_def:
                    true_def_line = guess_def
                    self.log(f"参数DEF修正: {filename}:{record['line']}→{guess_def} (via USE@{use_line})", "DEBUG")
                else:
                    self.log(f"USE变量[{use_var}]未找到DEF，保留原始行号", "WARN")

            # 行级 hint：真实定义行也打上
            self._record_method_hint(filename, true_def_line, origin_method)

            self._record_local_def(filename, def_line, var_id, origin_method, def_type, true_def_line)

        except Exception as e:
            self.log(f"处理DEF失败: {str(e)}", "ERROR")

    def handle_use(self, record):
        try:
            filename = self.get_filename(record['current_method'])
            var_id, line_num = record['var'], record['line']
            use_type = record.get('use_type', 0)
            target = record['target']

            # 行级 hint：use 行用 current_method
            self._record_method_hint(filename, line_num, record['current_method'])

            if use_type > 0 and target:
                target_file = self.get_filename(record['target'])
                if target_file != filename:
                    return
                self.call_args[target][use_type].append((line_num, var_id, filename))

            method_name = record['current_method'].split(':')[-1].split('(')[0].strip()
            var_name = var_id.split('@')[0]
            self.line_vars[line_num].add((method_name, var_name))

            visited = set()
            self._find_defs_iterative(filename, line_num, var_id, target, visited)

            use_node = self._get_latest_node(filename, line_num, method_hint=record['current_method'])
            if not use_node:
                use_node = self._create_node(filename, line_num, record['current_method'])
            use_node.uses[var_id]['count'] += 1
            use_node.uses[var_id]['type'] = use_type

            self.recent_uses.append((record['line'], record.get('use_type', 0), record['var'], filename))
        except Exception as e:
            self.log(f"处理USE失败: {str(e)}", "ERROR")

    # ---------------- 你的辅助逻辑保持不变 ----------------
    # def _add_both_edges(self, src_file, src_line, dst_file, dst_line, edge_type):
    #     if src_file != dst_file:
    #         return
    #     self._add_edge(src_file, src_line, dst_line, edge_type)

    def _find_previous_use(self, filename, current_line, target_type, max_lines=100):
        found = []
        for use in reversed(self.recent_uses):
            use_line, use_type, use_var, use_file = use
            if (use_file == filename and use_type == target_type and
                use_line < current_line and current_line - use_line <= max_lines):
                found.append({'var': use_var, 'line': use_line, 'file': use_file})
                return found[0]
        return None

    def _find_local_def(self, var_name, filename, max_line):
        try:
            if var_name in self.def_cache[filename]['local_vars']:
                defs = self.def_cache[filename]['local_vars'][var_name]
                for d in defs:
                    if d['line'] <= max_line:
                        return d['line']
            if var_name in self.def_cache[filename]['class_vars']:
                return self.def_cache[filename]['class_vars'][var_name]
            return None
        except KeyError:
            return None

    def _record_local_def(self, filename, line, var, method, def_type, trueline):
        self.def_cache[filename]['local_vars'][var].appendleft({
            'line': line,
            'type': def_type,
            'origin': method,
            'trueline': trueline
        })
        # 行级 hint：确保定义行有方法名
        self._record_method_hint(filename, trueline, method)
        def_node = self._get_latest_node(filename, trueline, method_hint=method)
        if not def_node:
            def_node = self._create_node(filename, trueline, method)
        def_node.defs[var]['count'] += 1
        def_node.defs[var].update({
            'type': def_type,
            'origin': method,
            'timestamp': self._get_new_timestamp()
        })

    def _find_defs_iterative(self, filename, start_line, var_id, target, visited):
        stack = [(filename, start_line, var_id, target, 0)]
        max_depth = 5
        while stack:
            current_file, current_line, current_var, current_target, depth = stack.pop()
            if depth > max_depth:
                continue
            cache_key = (current_file, current_line, current_var)
            if cache_key in visited:
                continue
            visited.add(cache_key)

            if current_var in self.def_cache[current_file]['class_vars']:
                def_line = self.def_cache[current_file]['class_vars'][current_var]
                # 行级 hint：类变量定义行也尽量标
                self._record_method_hint(current_file, def_line, self._peek_method(current_file))
                self._add_edge(current_file, current_line, def_line, 'class_use', current_var)
                continue

            if current_var in self.def_cache[current_file]['local_vars']:
                defs = self.def_cache[current_file]['local_vars'][current_var]
                if defs:
                    latest_def = defs[0]
                    # 行级 hint
                    self._record_method_hint(current_file, latest_def['trueline'], latest_def['origin'])
                    self._add_edge(current_file, current_line, latest_def['trueline'], 'data_flow', current_var)
                    if latest_def['type'] > 0 and depth < max_depth:
                        new_target = latest_def['origin']
                        stack.append((self.get_filename(new_target),
                                      latest_def['line'],
                                      current_var,
                                      new_target,
                                      depth + 1))

            if target:
                target_file = self.get_filename(target)
                if target_file in self.def_cache:
                    if current_var in self.def_cache[target_file]['class_vars']:
                        def_line = self.def_cache[target_file]['class_vars'][current_var]
                        self._record_method_hint(current_file, def_line, target)
                        self._add_edge(current_file, current_line, def_line, 'class_use', current_var)

    # ---------------- 保存/导出（原样） ----------------
    def save_results(self, output_dir):
        self.log(f"开始保存结果到 {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            for filename, graph in self.graphs.items():
                output = {
                    'filename': filename,
                    'nodes': {
                        n.id: {
                            'line': n.line,
                            'method': n.method,
                            'defs': {k: dict(v) for k, v in n.defs.items()},
                            'uses': dict(n.uses),
                            'calls': dict(n.calls),
                            'cfg_targets': [{"target": t, "count": c} for t, c in n.cfg.items()],
                            'is_entry': n.is_entry,
                            'is_exit': n.is_exit
                        } for ln, nodes in graph['nodes'].items() for n in nodes
                    },
                    'edges': {
                        f"{src_node_id}-{dst_node_id}": [
                            {'type': edge_type, 'var': var or None, 'count': count}
                            for (edge_type, var), count in edges.items()
                        ]
                        for (src_node_id, dst_node_id), edges in graph['edges'].items()
                    }
                }
                safe_name = self.sanitize_filename(filename).replace('/', '_')
                output_path = os.path.join(output_dir, f"{safe_name}.json")
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                self.log(f"成功保存 {filename} 的依赖图到 {output_path}")

            dataflow_dir = os.path.join(output_dir, "dataflow_graphs")
            self._generate_dataflow_dot(dataflow_dir)

            cfg_dir = os.path.join(output_dir, "cfg_graphs")
            self._generate_cfg_dot(cfg_dir)

        except Exception as e:
            self.log(f"保存结果失败: {str(e)}", "ERROR")
            raise

    def _generate_cfg_dot(self, output_dir="cfg_graphs"):
        try:
            os.makedirs(output_dir, exist_ok=True)
            for filename, graph in self.graphs.items():
                safe_name = self.sanitize_filename(filename).replace('/', '_')
                cfg_path = os.path.join(output_dir, f"{safe_name}_cfg.dot")

                node_id_map = {n.id: n for line_nodes in graph['nodes'].values() for n in line_nodes}

                cfg_edges = [
                    (src_id, dst_id, edge_type)
                    for (src_id, dst_id), edge_info in graph['edges'].items()
                    for (edge_type, var) in edge_info.keys()
                    if edge_type in ['cfg', 'call', 'return', 'entry']
                ]

                all_nodes = set()
                for src_id, dst_id, _ in cfg_edges:
                    all_nodes.update([src_id, dst_id])

                with open(cfg_path, 'w') as f:
                    f.write("digraph CFG {\n")
                    f.write('    graph [rankdir=LR];\n')
                    f.write('    node [shape=box];\n\n')

                    for node_id in all_nodes:
                        node = node_id_map.get(node_id)
                        if node:
                            label = f"L{node.line}"
                            if node.is_entry:
                                label += " (Entry)"
                            f.write(f'    "{node_id}" [label="{label}"];\n')

                    edge_styles = {
                        'cfg':    ('black', 'solid'),
                        'call':   ('blue',  'dashed'),
                        'return': ('green', 'dotted'),
                        'entry':  ('red',   'bold')
                    }

                    for src_id, dst_id, edge_type in cfg_edges:
                        if src_id in node_id_map and dst_id in node_id_map:
                            color, style = edge_styles.get(edge_type, ('black', 'solid'))  # ← 注意解包顺序
                            f.write(f'    "{src_id}" -> "{dst_id}" [color={color}, style={style}, label="{edge_type}"];\n')


                    f.write("}\n")
                self.log(f"生成CFG图: {cfg_path}")
        except Exception as e:
            self.log(f"生成CFG图失败: {str(e)}", "ERROR")

    def _generate_dataflow_dot(self, output_dir="dataflow_graphs"):
        try:
            os.makedirs(output_dir, exist_ok=True)
            for filename, graph in self.graphs.items():
                safe_name = self.sanitize_filename(filename).replace('/', '_')
                dot_path = os.path.join(output_dir, f"{safe_name}_dataflow.dot")
                node_id_map = {
                    n.id: n for line_num in graph['nodes'] for n in graph['nodes'][line_num]
                }
                dataflow_edges = []
                for (src_id, dst_id), edge_info in graph['edges'].items():
                    for (edge_type, var), count in edge_info.items():
                        if edge_type in ['data_flow', 'class_use']:
                            dataflow_edges.append((src_id, dst_id, edge_type, var, count))

                all_node_ids = set()
                for src_id, dst_id, *_ in dataflow_edges:
                    all_node_ids.update([src_id, dst_id])

                with open(dot_path, 'w') as f:
                    f.write("digraph DataFlow {\n")
                    f.write('    graph [rankdir=LR];\n')
                    f.write('    node [shape=box, style="rounded"];\n\n')

                    for node_id in all_node_ids:
                        node = node_id_map.get(node_id)
                        if node:
                            label = f"L{node.line}"
                            f.write(f'    "{node_id}" [label="{label}"];\n')

                    edge_styles = {
                        'data_flow': ('blue', 'solid', 'DF'),
                        'class_use': ('green', 'dashed', 'CU')
                    }

                    for src_id, dst_id, edge_type, var, count in dataflow_edges:
                        if src_id in node_id_map and dst_id in node_id_map:
                            color, style, tag = edge_styles.get(edge_type, ('black', 'solid', ''))
                            f.write(
                                f'    \"{src_id}\" -> \"{dst_id}\" [label=\"({count}x)\", color={color}, style={style}];\n'
                            )
                    f.write("}\n")
                self.log(f"生成数据流图: {dot_path}")
        except Exception as e:
            self.log(f"生成数据流图失败: {str(e)}", "ERROR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dependency Analyzer with Version Support')
    parser.add_argument('name', type=str)
    parser.add_argument('N', type=int, help='Version number (3-65)')
    args = parser.parse_args()

    analyzer = DependencyAnalyzer(debug=True)
    try:
        analyzer.process_logs('/tmp/instrument.log')
        analyzer.save_results(output_dir=f"graphs_all/{args.name}/{args.N}")
        print("分析完成，结果已保存")
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
