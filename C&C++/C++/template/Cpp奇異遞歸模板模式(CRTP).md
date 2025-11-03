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