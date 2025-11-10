# 正则表达式完整教程

## 目录
1. [正则表达式基础](#1-正则表达式基础)
2. [核心语法详解](#2-核心语法详解)
3. [实战案例](#3-实战案例)
4. [进阶技巧](#4-进阶技巧)
5. [速查表](#5-速查表)

---

## 1. 正则表达式基础

### 1.1 什么是正则表达式

正则表达式(Regular Expression, regex/regexp)是一种**文本模式描述工具**,用于:
- **搜索**: 在文本中查找符合模式的内容
- **匹配**: 验证文本是否符合特定格式
- **替换**: 批量修改符合模式的文本
- **提取**: 从文本中抽取特定信息

**核心思想**: 用特殊符号描述"一类字符串"的特征,而不是某个具体字符串。

### 1.2 正则引擎工作原理

#### 两种主要引擎类型

**1. NFA (非确定性有限自动机) - 大多数语言使用**
- **特点**: 以正则表达式为驱动,文本为被动匹配对象
- **匹配方式**:
  - 从左到右扫描文本
  - 遇到多个可能路径时,尝试第一个路径
  - 失败则回溯(backtrack)尝试其他路径
- **优点**: 支持反向引用、捕获组等高级特性
- **缺点**: 可能产生回溯,性能不稳定
- **使用语言**: Python, JavaScript, Java, Perl, .NET

**2. DFA (确定性有限自动机)**
- **特点**: 以文本为驱动
- **匹配方式**:
  - 从左到右扫描文本一次
  - 不回溯
- **优点**: 性能稳定,速度快
- **缺点**: 不支持反向引用等特性
- **使用工具**: grep, awk (部分模式)

#### NFA引擎匹配过程示例

**正则**: `a.*b`
**文本**: `axxxbyyy`

```
步骤1: a 匹配 'a' ✓
步骤2: .* 匹配 'xxxbyyy' (贪婪匹配)
步骤3: b 无法匹配 (已到文本末尾)
步骤4: 回溯, .* 吐出 'y' → 匹配 'xxxbyy'
步骤5: b 仍无法匹配
步骤6: 继续回溯, .* 吐出 'y' → 匹配 'xxxby'
步骤7: b 仍无法匹配
步骤8: 继续回溯, .* 匹配 'xxx'
步骤9: b 匹配 'b' ✓
最终匹配: 'axxxb'
```

### 1.3 应用场景

| 场景 | 示例 |
|------|------|
| **数据验证** | 验证邮箱、电话号码、IP地址格式 |
| **文本搜索** | 在代码中查找所有函数定义 |
| **数据提取** | 从日志中提取时间戳、错误码 |
| **格式转换** | CSV转JSON、数据格式标准化 |
| **文本清洗** | 删除多余空格、统一换行符 |
| **URL路由** | Web框架中的路由匹配 |
| **语法高亮** | 编辑器中的代码着色 |

---

## 2. 核心语法详解

### 2.1 字面量字符

**定义**: 在正则中直接代表自身的字符

```regex
hello     # 匹配 "hello"
123       # 匹配 "123"
abc123    # 匹配 "abc123"
```

**特殊情况**: 以下字符需要转义 (加反斜杠 `\`)
```
. ^ $ * + ? { } [ ] \ | ( )
```

**示例**:
```regex
\.        # 匹配字面量的点号 "."
\$100     # 匹配 "$100"
\(hello\) # 匹配 "(hello)"
```

### 2.2 元字符 (Metacharacters)

#### 2.2.1 基础元字符

| 元字符 | 含义 | 示例 | 匹配 | 不匹配 |
|--------|------|------|------|--------|
| `.` | 任意单个字符(除换行符) | `a.c` | abc, a1c, a@c | ac, abbc |
| `^` | 行首/字符串开始 | `^hello` | hello world | say hello |
| `$` | 行尾/字符串结束 | `world$` | hello world | world! |
| `\|` | 或 (alternation) | `cat\|dog` | cat, dog | catdog |
| `\` | 转义字符 | `\.` | . | 任意字符 |

#### 2.2.2 边界锚点

| 锚点 | 含义 | 示例 | 匹配 | 不匹配 |
|------|------|------|------|--------|
| `\b` | 单词边界 | `\bcat\b` | "cat", "the cat" | "catfish", "concat" |
| `\B` | 非单词边界 | `\Bcat\B` | "concatenation" | "cat", "the cat" |
| `^` | 字符串开始 | `^start` | start here | don't start |
| `$` | 字符串结束 | `end$` | the end | endless |

**单词边界示例**:
```regex
\bword\b 匹配:
  ✓ "word"
  ✓ "a word here"
  ✓ "word-play"
  ✗ "wording"
  ✗ "sword"
```

### 2.3 字符类 (Character Classes)

#### 2.3.1 基本字符类

**语法**: `[字符集合]`

```regex
[abc]      # 匹配 a 或 b 或 c
[a-z]      # 匹配任意小写字母
[A-Z]      # 匹配任意大写字母
[0-9]      # 匹配任意数字
[a-zA-Z]   # 匹配任意字母
[a-z0-9]   # 匹配字母或数字
```

**否定字符类**: `[^...]`
```regex
[^abc]     # 匹配除了 a, b, c 之外的任意字符
[^0-9]     # 匹配非数字字符
```

#### 2.3.2 预定义字符类

| 简写 | 等价 | 含义 |
|------|------|------|
| `\d` | `[0-9]` | 数字 |
| `\D` | `[^0-9]` | 非数字 |
| `\w` | `[a-zA-Z0-9_]` | 单词字符(字母、数字、下划线) |
| `\W` | `[^a-zA-Z0-9_]` | 非单词字符 |
| `\s` | `[ \t\n\r\f\v]` | 空白字符(空格、制表符、换行等) |
| `\S` | `[^ \t\n\r\f\v]` | 非空白字符 |

**示例**:
```regex
\d{3}-\d{4}        # 匹配 123-4567
\w+@\w+\.\w+       # 匹配简单邮箱 user@example.com
\s+                # 匹配一个或多个空白字符
```

### 2.4 量词 (Quantifiers)

#### 2.4.1 基本量词

| 量词 | 含义 | 示例 | 匹配 |
|------|------|------|------|
| `*` | 0次或多次 | `ab*c` | ac, abc, abbc, abbbc |
| `+` | 1次或多次 | `ab+c` | abc, abbc, abbbc (不匹配ac) |
| `?` | 0次或1次 | `ab?c` | ac, abc (不匹配abbc) |
| `{n}` | 恰好n次 | `a{3}` | aaa |
| `{n,}` | 至少n次 | `a{2,}` | aa, aaa, aaaa, ... |
| `{n,m}` | n到m次 | `a{2,4}` | aa, aaa, aaaa |

#### 2.4.2 贪婪 vs 非贪婪

**贪婪量词 (默认)**: 尽可能多地匹配
```regex
.*          # 贪婪匹配任意字符
.+          # 贪婪匹配至少一个字符
\d+         # 贪婪匹配数字
```

**非贪婪量词**: 加 `?` 后缀,尽可能少地匹配
```regex
.*?         # 非贪婪匹配
.+?         # 非贪婪匹配
\d+?        # 非贪婪匹配
{n,m}?      # 非贪婪匹配
```

**对比示例**:
```
文本: <div>content</div>

贪婪:     <.*>   匹配 "<div>content</div>" (整个字符串)
非贪婪:   <.*?>  匹配 "<div>" (第一个标签)
```

**详细过程**:
```
文本: axxxbyyy

贪婪 a.*b:
  a 匹配 'a'
  .* 匹配 'xxxbyyy' (尽可能多)
  回溯找到 b → 最终匹配 'axxxb'

非贪婪 a.*?b:
  a 匹配 'a'
  .*? 匹配 '' (尽可能少)
  b 不匹配 'x' → .*? 增加一个字符
  .*? 匹配 'x'
  b 不匹配 'x' → 继续...
  .*? 匹配 'xxx'
  b 匹配 'b' ✓ → 最终匹配 'axxxb'
```

### 2.5 分组与捕获

#### 2.5.1 捕获组 `(...)`

**作用**:
1. 将多个字符视为一个单元
2. 捕获匹配的内容供后续使用

**语法**:
```regex
(pattern)
```

**示例**:
```regex
(ab)+           # 匹配 ab, abab, ababab
(\d{3})-(\d{4}) # 匹配 123-4567,捕获两组数字
```

**捕获组编号**:
```regex
正则: (\d{4})-(\d{2})-(\d{2})
文本: 2025-11-10

捕获组:
  $0 或 \0: 2025-11-10 (完整匹配)
  $1 或 \1: 2025
  $2 或 \2: 11
  $3 或 \3: 10
```

#### 2.5.2 非捕获组 `(?:...)`

**作用**: 分组但不捕获,提高性能

```regex
(?:ab)+         # 匹配 ab, abab,但不捕获
(?:https?|ftp)://\w+  # 匹配协议但不捕获
```

**对比**:
```regex
捕获组:    (ab)+         # 捕获每次匹配的 ab
非捕获组:  (?:ab)+       # 不捕获,仅分组

性能: 非捕获组更快 (无需存储匹配内容)
```

#### 2.5.3 命名捕获组 `(?P<name>...)`

**Python语法**:
```python
(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})
```

**其他语言**:
```regex
# .NET, Perl
(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})

# JavaScript (ES2018+)
(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})
```

**使用示例** (Python):
```python
import re
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
text = '2025-11-10'
match = re.search(pattern, text)

print(match.group('year'))   # 2025
print(match.group('month'))  # 11
print(match.group('day'))    # 10
```

#### 2.5.4 反向引用

**定义**: 在正则中引用之前捕获的内容

**语法**:
```regex
\1, \2, \3...   # 引用第1、2、3个捕获组
```

**示例1: 匹配重复单词**
```regex
\b(\w+)\s+\1\b

匹配:
  ✓ "the the"
  ✓ "hello hello world"
  ✗ "the then"
```

**示例2: 匹配HTML标签**
```regex
<(\w+)>.*?</\1>

匹配:
  ✓ <div>content</div>
  ✓ <span>text</span>
  ✗ <div>content</span>
```

**示例3: 交换位置**
```regex
查找: (\w+),(\w+)
替换: \2,\1

输入: "Zhang,San"
输出: "San,Zhang"
```

### 2.6 断言 (Assertions)

#### 2.6.1 前瞻断言 (Lookahead)

**正向前瞻** `(?=...)`: 后面必须是...
```regex
\d+(?=元)       # 匹配后面跟"元"的数字
                # "100元" 中匹配 "100"

\w+(?=@)        # 匹配@前面的用户名
                # "user@example.com" 中匹配 "user"
```

**负向前瞻** `(?!...)`: 后面不能是...
```regex
\d+(?!元)       # 匹配后面不跟"元"的数字
                # "100" 匹配, "100元" 不匹配

\b\w+(?!ing\b)  # 匹配不以ing结尾的单词
```

#### 2.6.2 后顾断言 (Lookbehind)

**正向后顾** `(?<=...)`: 前面必须是...
```regex
(?<=\$)\d+      # 匹配$后面的数字
                # "$100" 中匹配 "100"

(?<=@)\w+       # 匹配@后面的域名部分
                # "user@example" 中匹配 "example"
```

**负向后顾** `(?<!...)`: 前面不能是...
```regex
(?<!\$)\d+      # 匹配前面不是$的数字
                # "100" 匹配, "$100" 不匹配
```

#### 2.6.3 断言综合示例

**匹配6-16位包含字母和数字的密码**:
```regex
^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,16}$

解析:
^                   # 开始
(?=.*[A-Za-z])      # 前瞻: 必须包含字母
(?=.*\d)            # 前瞻: 必须包含数字
[A-Za-z\d]{6,16}    # 6-16位字母或数字
$                   # 结束
```

**提取价格中的数字 (不含货币符号)**:
```regex
(?<=\$)\d+\.?\d*

匹配:
  "$100"    → "100"
  "$99.99"  → "99.99"
  "100"     → 不匹配
```

---

## 3. 实战案例

### 3.1 D2D数据格式转换案例详解

#### 3.1.1 需求分析

**源格式** (7字段):
```
0,0,gdma_0,15,ddr_1,W,4
```

**目标格式** (9字段):
```
0, 0, 0, gdma_0, 0, 15, ddr_1, W, 4
```

**任务**: 在第2和第5位置插入 `0, ` (源Die和目标Die)

#### 3.1.2 正则表达式设计

**查找模式**:
```regex
^(\d+),(\d+),(\w+),(\d+),(\w+),([RW]),(\d+)$
```

**分组解析**:
```
^                # 行首
(\d+)            # 组1: 时间戳 (一个或多个数字)
,                # 逗号
(\d+)            # 组2: 源节点
,                # 逗号
(\w+)            # 组3: 源IP类型 (单词字符)
,                # 逗号
(\d+)            # 组4: 目标节点
,                # 逗号
(\w+)            # 组5: 目标IP类型
,                # 逗号
([RW])           # 组6: 请求类型 (R或W)
,                # 逗号
(\d+)            # 组7: burst长度
$                # 行尾
```

**替换模式**:
```regex
$1, 0, $2, $3, 0, $4, $5, $6, $7
```

**替换逻辑**:
```
$1        → 时间戳 (原组1)
, 0,      → 插入源Die (固定值0)
$2        → 源节点 (原组2)
, $3,     → 源IP类型 (原组3)
0,        → 插入目标Die (固定值0)
$4        → 目标节点 (原组4)
, $5,     → 目标IP类型 (原组5)
$6,       → 请求类型 (原组6)
$7        → burst长度 (原组7)
```

#### 3.1.3 匹配过程详解

**示例输入**: `0,0,gdma_0,15,ddr_1,W,4`

**匹配步骤**:
```
步骤1: ^ 匹配行首
步骤2: (\d+) 捕获 "0" → $1 = "0"
步骤3: , 匹配逗号
步骤4: (\d+) 捕获 "0" → $2 = "0"
步骤5: , 匹配逗号
步骤6: (\w+) 捕获 "gdma_0" → $3 = "gdma_0"
步骤7: , 匹配逗号
步骤8: (\d+) 捕获 "15" → $4 = "15"
步骤9: , 匹配逗号
步骤10: (\w+) 捕获 "ddr_1" → $5 = "ddr_1"
步骤11: , 匹配逗号
步骤12: ([RW]) 捕获 "W" → $6 = "W"
步骤13: , 匹配逗号
步骤14: (\d+) 捕获 "4" → $7 = "4"
步骤15: $ 匹配行尾
```

**替换结果**:
```
$1, 0, $2, $3, 0, $4, $5, $6, $7
↓
0, 0, 0, gdma_0, 0, 15, ddr_1, W, 4
```

#### 3.1.4 VS Code操作步骤

1. **打开查找替换**: `Ctrl+H`
2. **启用正则**: 点击 `.*` 按钮
3. **输入查找**: `^(\d+),(\d+),(\w+),(\d+),(\w+),([RW]),(\d+)$`
4. **输入替换**: `$1, 0, $2, $3, 0, $4, $5, $6, $7`
5. **全部替换**: `Ctrl+Alt+Enter` 或点击"Replace All"

#### 3.1.5 Python实现

```python
import re

def convert_to_d2d_format(input_file, output_file):
    pattern = r'^(\d+),(\d+),(\w+),(\d+),(\w+),([RW]),(\d+)$'
    replacement = r'\1, 0, \2, \3, 0, \4, \5, \6, \7'

    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in:
                new_line = re.sub(pattern, replacement, line.strip())
                f_out.write(new_line + '\n')

# 使用
convert_to_d2d_format('data.txt', 'data_d2d.txt')
```

### 3.2 常见文本处理场景

#### 3.2.1 数据验证

**1. 邮箱验证**
```regex
^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$

示例:
  ✓ user@example.com
  ✓ john.doe@company.co.uk
  ✗ invalid@
  ✗ @example.com
```

**2. 中国手机号**
```regex
^1[3-9]\d{9}$

示例:
  ✓ 13812345678
  ✓ 19900001111
  ✗ 12345678901
  ✗ 1381234567
```

**3. IP地址**
```regex
^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$

示例:
  ✓ 192.168.1.1
  ✓ 10.0.0.255
  ✗ 256.1.1.1
  ✗ 192.168.1
```

**4. 日期格式 (YYYY-MM-DD)**
```regex
^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$

示例:
  ✓ 2025-11-10
  ✓ 2025-01-01
  ✗ 2025-13-01
  ✗ 2025-11-32
```

#### 3.2.2 数据提取

**1. 从日志提取时间戳**
```regex
\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}

日志: [2025-11-10 14:30:45] Error occurred
提取: 2025-11-10 14:30:45
```

**2. 提取URL中的域名**
```regex
(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]+)

输入: https://www.example.com/path
提取: example.com
```

**3. 提取代码中的函数名**
```regex
def\s+(\w+)\s*\(

代码: def calculate_sum(a, b):
提取: calculate_sum
```

**4. 提取括号内容**
```regex
\(([^)]+)\)

输入: The result (100) is correct.
提取: 100
```

#### 3.2.3 格式转换

**1. 驼峰命名转下划线**
```regex
查找: ([a-z])([A-Z])
替换: $1_$2

输入: getUserName
输出: get_user_name (需要后续转小写)
```

**Python完整实现**:
```python
import re

def camel_to_snake(name):
    s1 = re.sub('([a-z])([A-Z])', r'\1_\2', name)
    return s1.lower()

print(camel_to_snake('getUserName'))  # get_user_name
```

**2. 下划线转驼峰**
```python
import re

def snake_to_camel(name):
    return re.sub(r'_([a-z])', lambda m: m.group(1).upper(), name)

print(snake_to_camel('get_user_name'))  # getUserName
```

**3. 日期格式转换 (YYYY-MM-DD → DD/MM/YYYY)**
```regex
查找: (\d{4})-(\d{2})-(\d{2})
替换: $3/$2/$1

输入: 2025-11-10
输出: 10/11/2025
```

**4. 电话号码格式化**
```regex
查找: (\d{3})(\d{4})(\d{4})
替换: $1-$2-$3

输入: 13812345678
输出: 138-1234-5678
```

#### 3.2.4 文本清理

**1. 删除多余空格**
```regex
查找: \s+
替换: (单个空格)

输入: hello    world
输出: hello world
```

**2. 删除行首行尾空格**
```regex
查找: ^\s+|\s+$
替换: (空)

输入:   hello world
输出: hello world
```

**3. 删除空行**
```regex
查找: ^\s*\n
替换: (空)
```

**4. 统一换行符**
```regex
查找: \r\n|\r
替换: \n
```

### 3.3 VS Code查找替换实战

#### 3.3.1 基础操作

**快捷键**:
- `Ctrl+F`: 查找
- `Ctrl+H`: 替换
- `Alt+R`: 切换正则模式
- `Ctrl+Alt+Enter`: 全部替换

**选项按钮**:
- `Aa`: 大小写敏感
- `Ab`: 全词匹配
- `.*`: 正则表达式模式

#### 3.3.2 多行搜索

**启用**: 在查找框中按 `Ctrl+Enter` 换行

**示例**: 查找多行注释
```regex
/\*[\s\S]*?\*/

匹配:
/*
 * Multi-line comment
 * Example
 */
```

#### 3.3.3 多光标与正则

**场景**: 在每个匹配位置添加光标

1. `Ctrl+F` 打开查找
2. 输入正则: `\d+`
3. `Alt+Enter`: 选中所有匹配
4. 现在可以批量编辑

#### 3.3.4 实用案例

**案例1: 给JSON属性加引号**
```regex
查找: (\w+):
替换: "$1":

输入: {name: "John", age: 30}
输出: {"name": "John", "age": 30}
```

**案例2: 提取import语句中的模块名**
```regex
查找: ^import\s+(\w+)
替换: $1

输入: import numpy
输出: numpy
```

**案例3: 注释掉所有print语句**
```regex
查找: ^(\s*)print\(
替换: $1# print(

输入:     print("debug")
输出:     # print("debug")
```

### 3.4 Python re模块详解

#### 3.4.1 基本函数

**1. re.search() - 查找第一个匹配**
```python
import re

text = "The price is $100 and $200"
pattern = r'\$(\d+)'
match = re.search(pattern, text)

if match:
    print(match.group(0))  # $100 (完整匹配)
    print(match.group(1))  # 100 (第一组)
    print(match.start())   # 13 (起始位置)
    print(match.end())     # 17 (结束位置)
```

**2. re.match() - 从头匹配**
```python
text = "hello world"
re.match(r'hello', text)   # 匹配成功
re.match(r'world', text)   # 匹配失败 (不在开头)
```

**3. re.findall() - 查找所有匹配**
```python
text = "The prices are $100 and $200"
prices = re.findall(r'\$(\d+)', text)
print(prices)  # ['100', '200']
```

**4. re.finditer() - 返回迭代器**
```python
text = "The prices are $100 and $200"
for match in re.finditer(r'\$(\d+)', text):
    print(f"Found {match.group(1)} at position {match.start()}")

# 输出:
# Found 100 at position 15
# Found 200 at position 24
```

**5. re.sub() - 替换**
```python
text = "Hello World"
result = re.sub(r'World', 'Python', text)
print(result)  # Hello Python

# 使用函数替换
def upper_match(match):
    return match.group(0).upper()

text = "hello world"
result = re.sub(r'\w+', upper_match, text)
print(result)  # HELLO WORLD
```

**6. re.split() - 分割**
```python
text = "apple,banana;orange:grape"
parts = re.split(r'[,;:]', text)
print(parts)  # ['apple', 'banana', 'orange', 'grape']
```

#### 3.4.2 编译正则 (提高性能)

```python
import re

# 编译正则表达式
pattern = re.compile(r'\d+')

# 重复使用
text1 = "There are 123 apples"
text2 = "And 456 oranges"

print(pattern.search(text1).group())  # 123
print(pattern.search(text2).group())  # 456
```

**性能对比**:
```python
import re
import time

text = "test123" * 1000

# 不编译 (每次都编译)
start = time.time()
for _ in range(10000):
    re.search(r'\d+', text)
print(f"不编译: {time.time() - start:.3f}s")

# 预编译
pattern = re.compile(r'\d+')
start = time.time()
for _ in range(10000):
    pattern.search(text)
print(f"预编译: {time.time() - start:.3f}s")
```

#### 3.4.3 标志位 (Flags)

```python
import re

# 忽略大小写
re.search(r'hello', 'HELLO', re.IGNORECASE)
re.search(r'hello', 'HELLO', re.I)  # 简写

# 多行模式 (^和$匹配每行开头/结尾)
text = "line1\nline2"
re.findall(r'^line', text, re.MULTILINE)  # ['line', 'line']

# 点号匹配换行符
re.search(r'a.b', 'a\nb', re.DOTALL)

# 详细模式 (允许注释和空格)
pattern = re.compile(r'''
    \d{3}    # 区号
    -        # 分隔符
    \d{4}    # 号码
''', re.VERBOSE)

# 组合标志
re.search(r'hello', 'HELLO\nWORLD', re.I | re.M)
```

#### 3.4.4 完整示例: 日志解析

```python
import re
from datetime import datetime

log_pattern = re.compile(r'''
    (?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})  # 时间戳
    \s+
    \[(?P<level>\w+)\]                                    # 日志级别
    \s+
    (?P<message>.+)                                       # 消息
''', re.VERBOSE)

log_text = """
2025-11-10 14:30:45 [ERROR] Database connection failed
2025-11-10 14:30:50 [INFO] Retrying connection
2025-11-10 14:31:00 [ERROR] Connection timeout
"""

errors = []
for line in log_text.strip().split('\n'):
    match = log_pattern.search(line)
    if match and match.group('level') == 'ERROR':
        errors.append({
            'time': match.group('timestamp'),
            'message': match.group('message')
        })

for error in errors:
    print(f"{error['time']}: {error['message']}")

# 输出:
# 2025-11-10 14:30:45: Database connection failed
# 2025-11-10 14:31:00: Connection timeout
```

---

## 4. 进阶技巧

### 4.1 贪婪与非贪婪详解

#### 4.1.1 贪婪匹配原理

**NFA引擎策略**: "先匹配尽可能多,遇到失败再回溯"

**示例**:
```
文本: <div>hello</div><div>world</div>

正则: <div>.*</div>

匹配过程:
1. <div> 匹配 "<div>"
2. .* 贪婪匹配 "hello</div><div>world"
3. </div> 无法匹配 (已到末尾)
4. 回溯: .* 吐出 ">"
5. </div> 匹配成功
6. 最终匹配: "<div>hello</div><div>world</div>"
```

#### 4.1.2 非贪婪匹配原理

**策略**: "先匹配尽可能少,不够再增加"

**示例**:
```
文本: <div>hello</div><div>world</div>

正则: <div>.*?</div>

匹配过程:
1. <div> 匹配 "<div>"
2. .*? 匹配 "" (尽可能少)
3. </div> 不匹配 'h'
4. .*? 增加,匹配 "h"
5. </div> 不匹配 'e'
6. ... 继续 ...
7. .*? 匹配 "hello"
8. </div> 匹配成功
9. 最终匹配: "<div>hello</div>"
```

#### 4.1.3 何时使用非贪婪

**场景1: 提取HTML标签内容**
```regex
贪婪:   <div>.*</div>      # 匹配到最后一个</div>
非贪婪: <div>.*?</div>     # 匹配到第一个</div>
```

**场景2: 提取引号内容**
```regex
贪婪:   "(.*)"             # "a" and "b" → 匹配整个字符串
非贪婪: "(.*?)"            # "a" and "b" → 匹配 "a" 和 "b"
```

**场景3: 提取代码块**
```regex
非贪婪: ```.*?```          # 匹配第一个代码块
贪婪:   ```.*```           # 匹配到最后一个代码块
```

#### 4.1.4 性能考量

```python
import re
import time

text = 'a' * 10000 + 'b'

# 贪婪 (大量回溯)
start = time.time()
re.search(r'a.*a', text)
print(f"贪婪: {time.time() - start:.4f}s")

# 非贪婪 (更快)
start = time.time()
re.search(r'a.*?a', text)
print(f"非贪婪: {time.time() - start:.4f}s")

# 最优 (使用否定字符类)
start = time.time()
re.search(r'a[^a]*a', text)
print(f"否定类: {time.time() - start:.4f}s")
```

### 4.2 断言的高级应用

#### 4.2.1 密码强度验证

**要求**: 8-16位,必须包含大写、小写、数字

```regex
^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,16}$

解析:
^                   # 开始
(?=.*[a-z])         # 前瞻: 至少一个小写字母
(?=.*[A-Z])         # 前瞻: 至少一个大写字母
(?=.*\d)            # 前瞻: 至少一个数字
[a-zA-Z\d]{8,16}    # 8-16位字母或数字
$                   # 结束

测试:
✓ Abc12345
✓ MyPass123
✗ abc12345  (无大写)
✗ ABC12345  (无小写)
✗ MyPass    (无数字)
```

#### 4.2.2 匹配不包含特定词的行

**需求**: 匹配不包含"error"的日志行

```regex
^(?!.*error).*$

解析:
^               # 行首
(?!.*error)     # 负向前瞻: 整行不能包含error
.*              # 匹配整行
$               # 行尾

测试:
✓ "Info: Operation completed"
✓ "Warning: Low disk space"
✗ "Error: File not found"
✗ "Critical error occurred"
```

#### 4.2.3 提取价格但排除百分比

**需求**: 提取 $100 但不要 100%

```regex
\$\d+(?!\%)

测试:
✓ "$100 per item"     → 匹配 "$100"
✗ "100% discount"     → 不匹配
✓ "$50 (100% cotton)" → 匹配 "$50"
```

#### 4.2.4 千分位数字格式验证

**格式**: 1,234,567

```regex
^(?=\d{1,3}(,\d{3})*$)\d{1,3}(,\d{3})*$

或更简洁:
^\d{1,3}(,\d{3})*$

测试:
✓ 1,234
✓ 1,234,567
✗ 12,34        (第一组不是1-3位)
✗ 1,2345       (后续组不是3位)
```

### 4.3 性能优化

#### 4.3.1 避免灾难性回溯

**问题模式**:
```regex
(a+)+b

文本: aaaaaaaaaaaaaaaaaaaaaa (无b)
```

**回溯爆炸**:
```
尝试次数: 2^n (n为a的数量)
20个a → 1,048,576次尝试
30个a → 1,073,741,824次尝试 (超过10亿)
```

**解决方案**:
```regex
# 不好: (a+)+b
# 好:   a+b
# 好:   (?>a+)b  (原子组,不回溯)
```

#### 4.3.2 使用原子组 (?>...)

**原子组**: 一旦匹配成功,不允许回溯

```regex
(?>a+)b     # a+匹配后不回溯

文本: aaaaaab
- a+ 匹配 "aaaaaa"
- b 匹配 "b"
- 成功

文本: aaaaaac
- a+ 匹配 "aaaaaac"
- b 匹配失败
- 不回溯,直接失败
```

#### 4.3.3 优化技巧

**1. 使用具体字符类代替 `.`**
```regex
# 慢: .*@
# 快: [^@]*@
```

**2. 提取公共前缀**
```regex
# 慢: apple|application|apply
# 快: appl(?:e|ication|y)
```

**3. 使用锚点限制搜索**
```regex
# 慢: \d{3}-\d{4}
# 快: ^\d{3}-\d{4}$  (如果是完整匹配)
```

**4. 避免嵌套量词**
```regex
# 慢: (a*)*
# 快: a*
```

### 4.4 常见陷阱

#### 4.4.1 点号不匹配换行符

```python
import re

text = "line1\nline2"

# 失败
re.search(r'line1.line2', text)  # None

# 成功
re.search(r'line1.line2', text, re.DOTALL)  # Match
re.search(r'line1[\s\S]line2', text)        # Match
```

#### 4.4.2 忘记转义特殊字符

```regex
# 错误: 匹配 $100
\$100      # ✗ (100需要匹配恰好100)
\$100\.00  # ✗ (点号需要转义)

# 正确:
\$\d+\.\d{2}  # 匹配 $100.00
```

#### 4.4.3 贪婪量词导致的意外匹配

```regex
文本: <div>a</div><div>b</div>

# 意图: 匹配第一个div
# 错误:
<div>.*</div>  # 匹配整个字符串

# 正确:
<div>.*?</div>  # 匹配第一个
<div>[^<]*</div>  # 更快 (无回溯)
```

#### 4.4.4 字符类中的特殊规则

```regex
[.]       # 字符类中点号不需要转义,匹配字面量点
[^abc]    # ^在开头表示否定
[a-z]     # - 在中间表示范围
[-az]     # - 在开头/结尾表示字面量连字符
[a\-z]    # 转义的连字符
```

---

## 5. 速查表

### 5.1 元字符速查

| 元字符 | 含义 | 示例 |
|--------|------|------|
| `.` | 任意字符(除换行) | `a.c` → abc, a1c |
| `^` | 行首 | `^start` |
| `$` | 行尾 | `end$` |
| `*` | 0次或多次 | `ab*` → a, ab, abb |
| `+` | 1次或多次 | `ab+` → ab, abb |
| `?` | 0次或1次 | `ab?` → a, ab |
| `\|` | 或 | `cat\|dog` |
| `( )` | 捕获组 | `(ab)+` |
| `[ ]` | 字符类 | `[abc]` |
| `\` | 转义 | `\.` → 字面量点 |

### 5.2 字符类速查

| 简写 | 等价 | 含义 |
|------|------|------|
| `\d` | `[0-9]` | 数字 |
| `\D` | `[^0-9]` | 非数字 |
| `\w` | `[a-zA-Z0-9_]` | 单词字符 |
| `\W` | `[^a-zA-Z0-9_]` | 非单词字符 |
| `\s` | `[ \t\n\r\f\v]` | 空白字符 |
| `\S` | `[^ \t\n\r\f\v]` | 非空白字符 |
| `\b` | - | 单词边界 |
| `\B` | - | 非单词边界 |

### 5.3 量词速查

| 量词 | 含义 | 贪婪 | 非贪婪 |
|------|------|------|--------|
| `*` | 0次或多次 | `a*` | `a*?` |
| `+` | 1次或多次 | `a+` | `a+?` |
| `?` | 0次或1次 | `a?` | `a??` |
| `{n}` | 恰好n次 | `a{3}` | - |
| `{n,}` | 至少n次 | `a{2,}` | `a{2,}?` |
| `{n,m}` | n到m次 | `a{2,4}` | `a{2,4}?` |

### 5.4 断言速查

| 断言 | 类型 | 含义 | 示例 |
|------|------|------|------|
| `(?=...)` | 正向前瞻 | 后面是... | `\d(?=px)` → 100px中的100 |
| `(?!...)` | 负向前瞻 | 后面不是... | `\d(?!px)` |
| `(?<=...)` | 正向后顾 | 前面是... | `(?<=\$)\d+` → $100中的100 |
| `(?<!...)` | 负向后顾 | 前面不是... | `(?<!\$)\d+` |

### 5.5 标志位速查 (Python)

| 标志 | 简写 | 含义 |
|------|------|------|
| `re.IGNORECASE` | `re.I` | 忽略大小写 |
| `re.MULTILINE` | `re.M` | 多行模式(^$匹配每行) |
| `re.DOTALL` | `re.S` | .匹配换行符 |
| `re.VERBOSE` | `re.X` | 详细模式(允许注释) |
| `re.ASCII` | `re.A` | ASCII模式(\w只匹配ASCII) |

### 5.6 常用模式库

#### 5.6.1 数据验证

```regex
# 邮箱
^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$

# 手机号 (中国)
^1[3-9]\d{9}$

# 身份证
^[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]$

# URL
^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$

# IPv4
^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$

# 日期 YYYY-MM-DD
^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$

# 时间 HH:MM:SS
^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)$

# 十六进制颜色
^#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$

# 中文字符
[\u4e00-\u9fa5]
```

#### 5.6.2 提取模式

```regex
# HTML标签
<(\w+)[^>]*>(.*?)</\1>

# 引号内容
"([^"]*)"
'([^']*)'

# 括号内容
\(([^)]+)\)
\[([^\]]+)\]
\{([^}]+)\}

# 提取价格
\$(\d+\.?\d*)

# 提取IP:PORT
(\d+\.\d+\.\d+\.\d+):(\d+)

# 提取图片URL
!\[.*?\]\((.*?)\)

# 提取链接
\[([^\]]+)\]\(([^)]+)\)
```

#### 5.6.3 替换模式

```regex
# 删除HTML标签
查找: <[^>]+>
替换: (空)

# 删除多余空格
查找: \s+
替换: (单个空格)

# 删除空行
查找: ^\s*\n
替换: (空)

# 交换姓名位置
查找: (\w+),\s*(\w+)
替换: $2 $1

# 添加引号
查找: (\w+)
替换: "$1"
```

### 5.7 在线工具推荐

1. **regex101.com** - 最强大的在线测试工具
   - 实时匹配高亮
   - 详细解释每个步骤
   - 支持多种语言

2. **regexr.com** - 可视化工具
   - 实时可视化
   - 丰富的示例库

3. **debuggex.com** - 正则图形化
   - 将正则转换为铁路图
   - 理解复杂正则的好工具

---

## 附录: 不同语言的正则差异

### Python vs JavaScript

| 特性 | Python | JavaScript |
|------|--------|------------|
| 命名组 | `(?P<name>...)` | `(?<name>...)` |
| 非捕获组 | `(?:...)` | `(?:...)` ✓ 相同 |
| 引用组 | `\1` | `$1` (替换中) |
| 标志 | `re.I`, `re.M` | `/pattern/i`, `/pattern/m` |
| Unicode | `re.UNICODE` | `/pattern/u` |

### 示例对比

**Python**:
```python
import re
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})'
text = '2025-11'
match = re.search(pattern, text)
print(match.group('year'))  # 2025
```

**JavaScript**:
```javascript
const pattern = /(?<year>\d{4})-(?<month>\d{2})/;
const text = '2025-11';
const match = text.match(pattern);
console.log(match.groups.year);  // 2025
```

---

## 结语

正则表达式是文本处理的瑞士军刀。关键是:

1. **理解原理**: 了解NFA引擎的工作方式
2. **多练习**: 从简单模式开始,逐步复杂
3. **善用工具**: regex101等工具可视化匹配过程
4. **注意性能**: 避免灾难性回溯
5. **优先简洁**: 能用字符类就不用点号,能用锚点就加上

**学习路径建议**:
1. 掌握基础元字符和量词
2. 理解捕获组和反向引用
3. 学习断言的使用
4. 实践常见场景
5. 优化复杂正则的性能

祝你在正则表达式的世界里游刃有余!
