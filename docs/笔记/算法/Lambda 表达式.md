# Lambda 表达式

补题时别人使用lambda表达式感觉很好用，函数可以调用局部变量。

```c++
//离散化
vector<int> alls; 
sort(alls.begin(),alls.end());
int m = alls.erase(unique(alls.begin(),alls.end()),alls.end())-alls.begin();//最后范围是1-m
auto find = [alls](int x)->int{ //函数可以访问局部变量
    return  lower_bound(alls.begin(),alls.end(),x)-alls.begin()+1;
};
```

Lambda 表达式是 C++11 引入的一种匿名函数语法，可以快速定义和使用小型函数对象。

## 基本语法结构

```cpp
[捕获列表](参数列表) -> 返回类型 { 函数体 }
```

## 最简单的 Lambda 表达式

```cpp
auto sayHello = []() { 
    std::cout << "Hello Lambda!"; 
};
sayHello();  // 调用
```

## 带参数的 Lambda

```cpp
auto add = [](int a, int b) { 
    return a + b; 
};
std::cout << add(3, 5);  // 输出 8
```

## 捕获外部变量

### 1. 空捕获 `[]` - 不访问任何外部变量

```cpp
auto pure = [](int x) { return x * x; };  // 纯函数
cout << pure(5);  // 输出25
```

**适用场景**：  

- 纯计算逻辑，不依赖外部状态
- STL算法中的简单谓词（如`sort`的比较函数）

---

### 2. 值捕获 `[var]` - 复制变量

```cpp
int base = 100;
auto addBase = [base](int x) { return x + base; };
cout << addBase(5);  // 输出105

//默认情况下值捕获的变量是const的，使用mutable可以修改
int counter = 0;
auto inc = [counter]() mutable { 
    return ++counter;  // 可以修改counter的副本
};
```

**关键特性**：  
✅ 捕获时**固定变量值**（后续外部修改不影响）  
❌ 默认不能修改（除非加`mutable`）  

**离散化应用**：  

```cpp
// 离散化常用写法
sort(alls.begin(), alls.end());
auto find = [alls](int x) {  // 复制排序后的数组
    return lower_bound(alls.begin(), alls.end(), x) - alls.begin();
};
```

---

### 3. 引用捕获 `[&var]` - 直接操作原变量

```cpp
int cnt = 0;
vector<int> nums = {1,2,3};
for_each(nums.begin(), nums.end(), [&cnt](int x) {
    cnt += x;  // 直接修改外部cnt
});
cout << cnt;  // 输出6
```

**关键特性**：  
✅ 可修改原变量  
⚠️ 注意变量生命周期（Lambda可能比变量存活更久）  

**DFS应用**：  

```cpp
// DFS中的访问标记
vector<bool> vis(n); //需要修改
auto dfs = [&](int u) {  // 递归Lambda必须用引用捕获自身
    vis[u] = true;
    // ...
};

// 并查集带路径压缩
vector<int> fa(n);
auto find = [&fa](int x) {  // 只引用捕获fa
    return fa[x] == x ? x : fa[x] = find(fa[x]);
};
```

---

### 4. 隐式全捕获 `[=]` 和 `[&]`

```cpp
int a = 1, b = 2;
auto sum = [=]() { return a + b; };  // 值捕获所有
auto inc = [&]() { a++; b++; };      // 引用捕获所有
```

**使用建议**：  
🚫 慎用！明确写出需要捕获的变量更安全  
✅ 仅适合临时快速编码（如AtCoder的短代码）

---

### 5. 混合捕获 `[var1, &var2]`

```cpp
int config = 10, result = 0;
auto calc = [config, &result](int x) {
    result = x * config;  // config只读，result可写
};
```

---

### 📝 总结

| 捕获方式       | 语法     | 注意事项                  |
| -------------- | -------- | ------------------------- |
| 不捕获         | `[]`     | STL算法首选               |
| 值捕获         | `[var]`  | 最安全，推荐              |
| 引用捕获       | `[&var]` | 递归/修改外部变量时必须用 |
| 隐式全值捕获   | `[=]`    | 避免使用                  |
| 隐式全引用捕获 | `[&]`    | 快速编码时偶尔用          |
| 混合捕获       | `[a,&b]` | 精准控制捕获              |

## 返回类型推断

大多数情况下可以省略返回类型：

```cpp
auto square = [](int x) { return x * x; };  // 自动推断返回int
```

需要显式声明返回类型的情况：

```cpp
auto divide = [](double a, double b) -> double {
    if(b == 0) return 0;
    return a / b;
};
```

## 立即调用的 Lambda

```cpp
int result = [](int x) { return x * x; }(5);  // result = 25
```
