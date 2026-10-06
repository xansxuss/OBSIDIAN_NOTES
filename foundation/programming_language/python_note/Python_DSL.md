---
title: Python DSL 設計教學
source: 
author: 
published: 
created: 2026-10-06
description: 以運算子多載、表達式樹、遞迴下降 parser 與 ast 模組，說明 Python 內部與外部 DSL 的設計方法與陷阱。
categorization: foundation/programming_language/python_note
tags:
  - Python
  - DSL
  - 運算子多載
  - AST
  - Parser
  - 設計模式
---

# Python DSL 設計教學

> 建議存放路徑：`foundation/programming_language/python_note/`

## 1. 核心觀念

- DSL（Domain-Specific Language）把「怎麼做」藏起來，讓程式碼讀起來像在描述「做什麼」。
- 條件變成**物件**後，可以被列印、最佳化、序列化，或轉譯成 SQL、GPU kernel 等其他目標。
- 設計流程：先寫出希望使用者怎麼寫 → 選擇內部或外部 DSL → 實作。

| 類型 | 說明 | 範例 |
|---|---|---|
| 內部 DSL（Embedded） | 用 Python 語法特性包裝出像新語言的 API | SQLAlchemy、pandas、PyTorch `nn.Sequential` |
| 外部 DSL（External） | 自訂語法，需自己寫 tokenizer 與 parser | `lark`、`ply`、手寫遞迴下降 |

## 2. 內部 DSL 五大技巧

### 2.1 運算子多載

| 運算子 | 方法 | 運算子 | 方法 |
|---|---|---|---|
| `a + b` | `__add__` | `a > b` | `__gt__` |
| `a \| b` | `__or__` | `a == b` | `__eq__` |
| `a & b` | `__and__` | `~a` | `__invert__` |
| `a >> b` | `__rshift__` | `a / b` | `__truediv__` |
| `a @ b` | `__matmul__` | `a[i]` | `__getitem__` |

- 常數在左邊（`1 + a`）時，Python 會先試 `int.__add__`，失敗才呼叫 `a.__radd__(1)`，所以必須實作 `__radd__` 系列。
- 例子：用 `|` 串接影像前處理步驟 `normalize | clip | to_chw`，每次回傳新的 `Pipeline`，避免共用狀態。

### 2.2 `__getattr__`：動態產生語法

```python
class Path:
    def __init__(self, parts=()):
        self._parts = tuple(parts)          # tuple 不可變，避免共用狀態

    def __getattr__(self, name):
        # 只有「正常查找失敗」才會進來
        if name.startswith("_"):
            raise AttributeError(name)      # 擋掉 __deepcopy__ 等內部查詢，避免無限遞迴
        return Path(self._parts + (name,))

    def __repr__(self):
        return "/".join(self._parts)

print(Path().models.yolov8.engine)          # models/yolov8/engine
```

### 2.3 Context manager：表達範圍

```python
class Scope:
    _stack = []                             # 類別層級堆疊，所有 Scope 共用

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        Scope._stack.append(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        Scope._stack.pop()                  # 不論是否有例外都會彈出
        return False                        # 不吞掉例外

    @staticmethod
    def full(name):
        return "/".join(Scope._stack + [name])
```

### 2.4 Decorator：宣告式註冊（registry 模式）

```python
_REGISTRY = {}

def register(name):
    def deco(func):
        _REGISTRY[name] = func              # 副作用：登記到表中
        return func                         # 原函式原封不動回傳
    return deco

@register("normalize")
def normalize(x):
    return [v / 255.0 for v in x]

def build(names):
    funcs = [_REGISTRY[n] for n in names]   # 名稱可來自 JSON / YAML 設定檔
    def run(data):
        for f in funcs:
            data = f(data)
        return data
    return run
```

- PyTorch、mmdetection 的 registry 都是這個模式。

### 2.5 Fluent interface 與延遲求值

- 每個方法回傳**新物件**並累積操作，最後 `run()` 才一次執行。
- 好處：執行前可分析並重排操作，這是查詢最佳化器的雛形。

## 3. 實戰：表達式樹 + 查詢 DSL

目標語法：

```python
q = (query(rows)
     .where((col("score") > 60) & (col("age") < 30))
     .order_by("score", desc=True)
     .select("name", "score")
     .limit(2))
```

### 3.1 表達式樹

```python
class Expr:
    """基底類別。比較運算回傳 BinOp 節點，而不是 bool。"""
    def eval(self, row):
        raise NotImplementedError

    def __gt__(self, o): return BinOp(">", self, to_expr(o))
    def __lt__(self, o): return BinOp("<", self, to_expr(o))
    def __eq__(self, o): return BinOp("==", self, to_expr(o))
    # and / or / not 無法多載，改用 & | ~
    def __and__(self, o): return BinOp("and", self, to_expr(o))
    def __or__(self, o):  return BinOp("or", self, to_expr(o))
    def __invert__(self): return Not(self)
    # 反向版本，支援 10 + col("a")
    def __add__(self, o):  return BinOp("+", self, to_expr(o))
    def __radd__(self, o): return BinOp("+", to_expr(o), self)

class Const(Expr):
    def __init__(self, value): self.value = value
    def eval(self, row): return self.value
    def __repr__(self): return repr(self.value)

class Col(Expr):
    def __init__(self, name): self.name = name
    def eval(self, row): return row[self.name]    # 求值時才從 row 取值
    def __repr__(self): return self.name

_OPS = {
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "and": lambda a, b: a and b, "or": lambda a, b: a or b,
    "+": lambda a, b: a + b,
}

class BinOp(Expr):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right
    def eval(self, row):
        # 遞迴求值：先算左右子樹，再套用運算子
        return _OPS[self.op](self.left.eval(row), self.right.eval(row))
    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

class Not(Expr):
    def __init__(self, child): self.child = child
    def eval(self, row): return not self.child.eval(row)

def to_expr(x):
    return x if isinstance(x, Expr) else Const(x)    # 普通值包成 Const

def col(name):
    return Col(name)
```

- 重點：`col("score") > 60` 沒有比較任何東西，而是**建出樹節點**。
- 定義 `__eq__` 後 `__hash__` 會變 `None`，`Expr` 不能當 dict key，這是刻意取捨。

### 3.2 Query 物件（延遲求值）

```python
class Query:
    def __init__(self, rows, ops=()):
        self.rows, self.ops = rows, ops

    def _add(self, op):
        return Query(self.rows, self.ops + (op,))   # 回傳新物件，原物件不變

    def where(self, expr):             return self._add(("where", expr))
    def order_by(self, name, desc=False): return self._add(("order", name, desc))
    def select(self, *names):          return self._add(("select", names))
    def limit(self, n):                return self._add(("limit", n))

    def run(self):
        data = list(self.rows)                      # 複製，避免改到原始資料
        for op in self.ops:                         # 依加入順序執行
            if op[0] == "where":
                data = [r for r in data if op[1].eval(r)]
            elif op[0] == "order":
                data.sort(key=lambda r: r[op[1]], reverse=op[2])
            elif op[0] == "select":
                data = [{k: r[k] for k in op[1]} for r in data]
            elif op[0] == "limit":
                data = data[:op[1]]
        return data

def query(rows):
    return Query(rows)
```

## 4. 外部 DSL：Tokenizer + 遞迴下降 Parser

目標：把字串 `score > 60 and (age < 30 or name == "Amy")` 解析成上一節的 `Expr` 樹，兩種 DSL 共用同一個執行核心。

### 4.1 文法（EBNF）

優先權由低到高，**文法的層次就是優先權**：

```
or_expr  := and_expr ("or" and_expr)*
and_expr := not_expr ("and" not_expr)*
not_expr := "not" not_expr | cmp
cmp      := arith (("=="|"!="|">="|"<="|">"|"<") arith)?
arith    := term (("+"|"-") term)*
term     := factor (("*"|"/") factor)*
factor   := NUM | STR | NAME | "(" or_expr ")" | "-" factor
```

### 4.2 Tokenizer 重點

- 逐字元掃描，產生 `(類型, 值)` 串列，結尾補 `("EOF", None)` 簡化判斷。
- 兩字元運算子（`==`、`>=` 等）要**先於**單字元檢查。
- `and` / `or` / `not` 為關鍵字，其餘識別字為 `NAME`。
- 無法辨識的字元要丟 `SyntaxError` 並附位置。

### 4.3 Parser 重點

- 每條文法規則對應一個方法，這就是「遞迴下降」。
- `while` 迴圈產生**左結合**：`a or b or c` = `(a or b) or c`。
- 括號內呼叫 `or_expr()` 回到最低優先權。
- 一元負號可轉成 `0 - x`。

```python
def or_expr(self):
    node = self.and_expr()
    while self.peek()[0] == "or":           # 左結合
        self.advance()
        node = BinOp("or", node, self.and_expr())
    return node

def factor(self):
    kind, val = self.advance()
    if kind in ("NUM", "STR"):
        return Const(val)
    if kind == "NAME":
        return Col(val)
    if kind == "OP" and val == "(":
        node = self.or_expr()               # 括號內回到最低優先權
        if not self.match_op(")"):
            raise SyntaxError("缺少右括號 )")
        return node
    raise SyntaxError(f"非預期的 token：{val!r}")
```

- 驗證優先權：`parse("1 + 2 * 3")` 應得 `(1 + (2 * 3))`。
- 文法變大時改用 [[lark]]：直接貼 EBNF 即可產生 parser，並附錯誤位置。

## 5. 進階：`ast` 模組與 Triton 類 DSL

Triton、Taichi、Numba `@jit` 的共同手法：**取得函式原始碼 → 解析成 Python AST → 轉譯成目標程式碼**。

```python
import ast
import inspect

def show_tree(func):
    src = inspect.getsource(func)           # 取得原始碼字串
    print(ast.dump(ast.parse(src), indent=2))

class ToC(ast.NodeVisitor):
    """把簡單 Python 運算式轉成 C 運算式。"""
    _SYM = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def visit_BinOp(self, node):
        l, r = self.visit(node.left), self.visit(node.right)
        return f"({l} {self._SYM[type(node.op)]} {r})"

    def visit_Name(self, node):
        return node.id

    def visit_Constant(self, node):
        return repr(node.value)

expr = ast.parse("a * 2 + b", mode="eval").body
print(ToC().visit(expr))                    # ((a * 2) + b)
```

- 概念上等同 C++ 的 [[C++ expression template]]（運算子多載建樹、編譯期最佳化），只是 Python 在執行期做。
- 延伸閱讀：[[Triton jit 原理]]、[[Python ast 模組]]。

## 6. 設計原則與常見陷阱

| 問題 | 對策 |
|---|---|
| `and` / `or` / `not` 無法多載 | 改用 `&` `\|` `~`，或改走外部 DSL |
| `&` 優先權高於比較 | `col("a") > 1 & col("b") < 2` 會出錯，必須加括號 |
| `__eq__` 被覆寫 | `__hash__` 變 `None`，不能當 dict key，`x in list` 也可能出現意外結果 |
| `__bool__` 誤用 | `if col("a") > 1:` 會悄悄成立，應在 `Expr.__bool__` 直接 `raise TypeError("請用 & 與 \|")` |
| 可變狀態 | fluent 方法回傳新物件，不修改 `self` |
| 錯誤訊息 | 外部 DSL 一定附位置；內部 DSL 在建構時檢查型別 |
| 過度設計 | 只用一次的邏輯，直接寫函式更清楚 |

### 設計檢查清單

1. 語法是否符合領域專家的習慣？
2. 能否被列印、序列化、除錯？（`__repr__` 一定要寫）
3. 求值與建構是否分離？（才能最佳化與轉譯）
4. 出錯時使用者能否定位問題？

## 7. 練習題

1. 為 `Expr` 加上 `__bool__` 並丟出有意義的錯誤。
2. 加入 `col("name").isin([...])`，外部 DSL 支援 `name in ("Amy", "Bob")`。
3. 寫 `to_sql(expr)`，把表達式樹輸出成 SQL `WHERE` 字串。
4. 把影像前處理 `Pipeline` 改成外部 DSL：`"resize(640,640) | normalize | to_chw"`，並用 registry 對應函式。
5. 用 `ast` 寫轉譯器，把 `a * 2 + b` 輸出成 C++ `for` 迴圈 kernel。

## 相關筆記

- [[Python 運算子多載]]
- [[Python Decorator]]
- [[遞迴下降 Parser]]
- [[C++ expression template]]
- [[Python ast 模組]]
