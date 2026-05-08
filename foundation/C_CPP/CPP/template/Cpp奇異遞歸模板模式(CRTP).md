前言

C++ 奇異遞迴模板模式（Curiously Recurring Template Pattern，簡稱 CRTP）是一種模板程式設計技巧。
其特點是「派生類別將自己作為基底類別模板的模板參數」，讓基底類別能夠在編譯期以靜態多型的方式使用派生類別的屬性與方法，從而達到靜態多型（static polymorphism）與編譯期行為決定的效果。

這種技術能在不使用虛函式的情況下實現多型行為，減少執行期開銷，但相對地，靈活性不如執行期多型，且模板錯誤訊息往往較難除錯。

其一般形式如下：

``` cpp
template <typename T>
class Base {
};

class Derived : public Base<Derived> {
    // Derived 作為 Base 的模板參數，同時又繼承自 Base
    // 這樣 Base 內部即可將 this 轉換為 T 型別，從而呼叫派生類別的介面
};
```

應用場景
1. 靜態多型（Static Polymorphism）

透過基底類別的介面呼叫派生類別的實作：

``` cpp
template <typename Derived>
class Shape {
public:
    void draw() {
        static_cast<Derived *>(this)->drawImpl();
    }
};

class Circle : public Shape<Circle> {
public:
    void drawImpl() { }
};
```

在 C++23 中引入的 Deducing this 特性（顯式 this 參數）可進一步簡化 CRTP 的寫法，
不再需要顯式將派生類別作為模板實參：

``` cpp
class Shape {
public:
	// Deducing this
	void draw(this auto& self) {  
		self.drawImpl();
	}
};

class Circle : public Shape {
public:
	void drawImpl() {}
};
```

2. mixin（功能注入）

mixin 是一種程式碼重用（code reuse）模式，
可透過繼承或組合的方式將功能「混入」到一個類別中，
讓類別獲得額外的行為或屬性。這種概念可類比為 Python 的裝飾器（decorator）。

mixins 通常表示 has-a（擁有某功能） 的關係，而非 is-a（是一種） 的關係。

``` cpp
template <typename Derived>
class CountableMixin {
private:
    // C++17 起允許 inline 變數初始化
    static inline int counter{ 0 };
public:
    CountableMixin() { ++counter; }
    ~CountableMixin() { --counter; }
    static int count() { return counter; }
};

class MyClass : public CountableMixin<MyClass> {
    // 自動擁有計數功能
};
```

另一個例子：比較運算 mixin。

``` cpp
template <typename T>
struct Comparable {
	friend bool operator==(const T& a, const T& b) {
		return a.compare(b) == 0;
	}
	friend bool operator>(const T& a, const T& b) {
		return a.compare(b) > 0;
	}
    // ... ...
};

// 多重繼承可混合多種能力
class MyInt : public Comparable<MyInt> {
public:
	MyInt(int v) : value(v) {}
	// 定義 compare 之後，即自動擁有一系列比較運算子
	int compare(const MyInt& other) const {
		return value - other.value;
	}
private:
	int value;
};
```

3. 模板方法模式（Template Method Pattern）

模板方法是一種行為設計模式，
在父類中定義演算法骨架（流程），
但將部分步驟延後到子類別實作。

``` cpp
template <typename T>
class Base {
public:
    void algorithm() {
        static_cast<T*>(this)->step2();
    }
    void step1() {}
};

class Derived : public Base<Derived> {
public:
    void step2() {}
};
```

除此之外，工廠模式（Factory Pattern）、策略模式（Strategy Pattern） 等多種經典設計模式，也都能透過 CRTP 靜態化實現。

4. std::enable_shared_from_this

C++ 標準庫中的 std::enable_shared_from_this 是一個模板類別，
允許物件安全地取得指向自身的 shared_ptr，
避免因從原始指標（raw pointer）建立多個 shared_ptr 而導致的重複釋放問題。

其原理是內部維護一個 weak_ptr，
當以 shared_ptr 建構物件時，該 weak_ptr 會被初始化，
後續即可透過 shared_from_this() 安全地取得同一份共享控制區。

``` cpp
#include <memory>
#include <iostream>

struct MyItem : public std::enable_shared_from_this<MyItem>
{
	int value;
	std::shared_ptr<MyItem> getSharedPtr()
	{
		return shared_from_this();
	}
};

int main()
{
	auto s1 = std::make_shared<MyItem>();
	auto s2 = s1->getSharedPtr();

	s1->value = 3;
	std::cout << "s2 value = " << s2->value << std::endl;
	return 0;
}
```

✅ 總結
CRTP 是一種結合模板與繼承的強大設計手法，
可在不使用虛函式的情況下實現多型行為，並且在編譯期完成優化。
它是許多高效能 C++ 函式庫（例如 Eigen、Boost、STL）的底層實作基石之一。

### CRTP mixin and python界面裝飾器

1) CRTP mixin（C++）：核心概念與實作範例

CRTP（Curiously Recurring Template Pattern）是 Base<Derived> 的形式，透過 static_cast 把靜態型別資訊帶進基底實作，達到編譯期多型與 mixin 的效果 — 沒有虛擬表（vtable）開銷。

基本範例 — 提供 serialize() 功能的 mixin：

``` cpp
// CRTP mixin: 可重用的序列化工具
#include <iostream>
#include <string>

template<typename Derived>
struct ToStringMixin {
    // 呼叫 Derived::to_string_impl()（由派生實作），在基底提供公共 API
    std::string to_string() const {
        // static_cast 安全地把 this 轉回 Derived*
        return static_cast<const Derived*>(this)->to_string_impl();
    }
};

struct Point : ToStringMixin<Point> {
    int x, y;
    Point(int x_, int y_) : x(x_), y(y_) {}
    std::string to_string_impl() const {
        return "Point(" + std::to_string(x) + "," + std::to_string(y) + ")";
    }
};

int main() {
    Point p{1,2};
    std::cout << p.to_string() << "\n"; // => Point(1,2)
}
```

重點／優勢：

	- 無虛擬呼叫，inlineable，效能好。
	- 很適合把小功能（logging、to_string、comparable、iterator helper）做成 mixin。

常見進階需求與技巧：
	1. 多個 mixin 串接：struct A : Mixin1<A>, Mixin2<A> { ... };
	2. 編譯期檢查派生類別是否提供必要函式：用 requires（C++20）或 detection idiom（SFINAE）做 static_assert。
	3. 避免二義性：當多個 mixin 提供相同名字時需解決命名衝突（qualify 或把實作放到不同命名空間）。

檢查派生要有 to_string_impl() 的範例（C++20 concepts）：

``` cpp
#include <concepts>
#include <string>

template<typename D>
concept HasToStringImpl = requires(const D& d) {
    { d.to_string_impl() } -> std::convertible_to<std::string>;
};

template<HasToStringImpl Derived>
struct ToStringMixin {
    std::string to_string() const {
        return static_cast<const Derived*>(this)->to_string_impl();
    }
};
```

踩雷提醒：

	- CRTP 非常強，但會增加 template 錯誤訊息複雜度；別把過多責任塞進單一 mixin。
	- 若 mixin 需要狀態，務必知道每個派生都會各自擁有該狀態（不會共享）。

2) Python 的「介面 / 裝飾器」：幾種實用模式

Python 沒有 compile-time 多型，但動態性強，可用class decorator或metaclass做介面檢查、或用function/class decorator注入 mixin 行為。

情境 A：類別層級檢查 — 確保類別實作指定方法（interface enforcement）

用 class decorator 檢查必須實作的方法：

``` python
# interface_decorator.py
import inspect
from functools import wraps

def requires_methods(*methods):
    def decorator(cls):
        missing = [m for m in methods if not any(m in B.__dict__ for B in cls.__mro__)]
        if missing:
            raise TypeError(f"Class {cls.__name__} missing required methods: {missing}")
        return cls
    return decorator

# 用法
@requires_methods('to_string', 'serialize')
class MyThing:
    def to_string(self): return "hi"
    def serialize(self): return b"hi"
```

注意：這個檢查是在 class 定義時（import/裝載時）就會執行，很像靜態檢查。

情境 B：自動注入 mixin 方法（類似 C++ mixin，但動態）

把 mixin 的方法注入到目標類別（不改變 MRO，只是複製函式）：

``` python
def inject_mixin(mixin):
    def decorator(cls):
        for name, attr in mixin.__dict__.items():
            if callable(attr) and not name.startswith('__'):
                # 不覆寫已存在的方法
                if name not in cls.__dict__:
                    setattr(cls, name, attr)
        return cls
    return decorator

# 範例 mixin
class PrintableMixin:
    def to_string(self):
        return f"<{self.__class__.__name__}>"

# 使用
@inject_mixin(PrintableMixin)
class Widget:
    pass

w = Widget()
print(w.to_string())  # => <Widget>
```

情境 C：用 decorator 包裝類別以加上通用行為（logging、thread-safety）

``` python
import threading, functools

def synchronized(cls):
    """在每個方法前自動加鎖（簡化示例）"""
    lock = threading.RLock()
    for name, fn in list(cls.__dict__.items()):
        if callable(fn) and not name.startswith('__'):
            @functools.wraps(fn)
            def wrapped(self, *a, __fn=fn, **kw):
                with lock:
                    return __fn(self, *a, **kw)
            setattr(cls, name, wrapped)
    return cls

@synchronized
class Counter:
    def __init__(self):
        self.n = 0
    def inc(self):
        self.n += 1
```

進階：用 inspect.signature 做簽名檢查，或使用 abc.ABC + @abstractmethod 做更正式的介面檢查。

3) 把 CRTP 想法映射到 Python（模擬靜態多型）

如果你想要在 Python 裡模仿「CRTP 的風格」── 在基底提供一組 API，然後呼叫子類的具體函式（而不額外的抽象基類），可以用 class decorator 同步注入一個 wrapper 呼叫子類實作：

``` python
def crtp_like(method_name):
    """在基底類別注入一個呼叫子類實作的共通方法"""
    def decorator(base_cls):
        def common(self, *a, **k):
            impl = getattr(self, method_name, None)
            if impl is None:
                raise NotImplementedError(f"{self.__class__.__name__} must implement {method_name}")
            return impl(*a, **k)
        setattr(base_cls, method_name.replace('_impl',''), common)
        return base_cls
    return decorator

@crtp_like('to_string_impl')
class Base:
    pass

class Real(Base):
    def to_string_impl(self):
        return "i am real"

r = Real()
print(r.to_string())  # => i am real
```

這本質上是動態注入，但語意上很像 CRTP：基底提供公共 API，實際實作由子類提供。

4) 同場加映：把 C++ CRTP 與 Python 接面接軌（pybind11 小貼士）

你若要把 C++ CRTP 類別暴露給 Python（使用 pybind11），記住：

CRTP 產生的成員函式通常是非虛擬的，pybind11 若要覆寫 Python 層 方法，需要 C++ 有虛擬函式（或透過 trampoline wrapper）。

若你只是把 to_string() 這類非虛擬函式當作 utility 暴露，直接綁定即可；但若希望 Python override，那得改用 virtual 或用委派（delegate）設計。

簡單建議：在 C++ 端把「可被 Python override 的行為」抽成純虛擬介面，其他 non-virtual 的 mixin 照常用 CRTP。

5) 最後總結（快速筆記）

CRTP：編譯期多型、低開銷、適合 mixin，但 template error 與可讀性較差。用 concepts / detection idiom 加強診斷。

Python 裝飾器：在類別定義時做檢查、注入或自動包裝，非常靈活。若要強制約束，prefer abc 或 class decorator 做早期錯誤（fail-fast）。

如果要跨語言：暴露給 Python 時，注意哪些方法要 virtual / non-virtual；pybind11 需要 trampoline 類來支援 Python 覆寫。