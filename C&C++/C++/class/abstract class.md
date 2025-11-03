## abstract class

在 C++ 裡，abstract class（抽象類別） 是一種不能被直接實例化（instantiate）的類別，用來作為介面（interface）或基底類別（base class），提供其他類別繼承與實作的框架。

🧩 一、定義方式

抽象類別的關鍵在於「純虛函式（pure virtual function）」。

``` cpp
class Shape {
public:
    // 純虛函式：沒有實作，= 0 表示必須由子類別實作
    virtual void draw() = 0;

    // 抽象類別可以有普通成員或虛函式
    virtual double area() const { return 0.0; }

    // 虛擬解構函式，避免多型刪除時記憶體洩漏
    virtual ~Shape() {}
};
```

上面這個 Shape 類別就是一個抽象類別，因為它有一個 pure virtual function：

``` cpp
virtual void draw() = 0;
```

🧬 二、使用方式

抽象類別不能直接建立物件：

``` cpp
Shape s;  // ❌ 錯誤：抽象類別不能被實例化
```

只能透過「繼承並實作純虛函式」的子類別來使用：

``` cpp
class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    void draw() override { std::cout << "Drawing Circle\n"; }
    double area() const override { return 3.14159 * radius * radius; }
};
```

🧪 三、使用範例

``` cpp
#include <iostream>
#include <vector>
#include <memory>

class Shape {
public:
    virtual void draw() = 0;             // 純虛函式
    virtual double area() const = 0;     // 純虛函式
    virtual ~Shape() = default;
};

class Circle : public Shape {
    double r;
public:
    Circle(double radius) : r(radius) {}
    void draw() override { std::cout << "畫圓\n"; }
    double area() const override { return 3.14 * r * r; }
};

class Rectangle : public Shape {
    double w, h;
public:
    Rectangle(double width, double height) : w(width), h(height) {}
    void draw() override { std::cout << "畫矩形\n"; }
    double area() const override { return w * h; }
};

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(3.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 2.0));

    for (auto& s : shapes) {
        s->draw();
        std::cout << "面積: " << s->area() << "\n";
    }
}
```

⚙️ 四、抽象類別的用途

- 定義統一介面：讓不同類型的物件（如 Circle / Rectangle）能以多型方式操作。
- 強制子類別實作特定功能。
- 支援多型（Polymorphism）與動態綁定（Dynamic Binding）。
- 設計模式基礎：像是 Template Method、Strategy、Factory 都常以抽象類別為基底。

💡 延伸：純虛析構函式
有時會這樣寫：

``` cpp
class Base {
public:
    virtual ~Base() = 0; // 純虛析構函式
};
Base::~Base() {} // 必須提供實作！
```

👉 這樣 Base 仍是抽象類別，但確保子類別被刪除時能正確呼叫析構函式。
