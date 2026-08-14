std::mutex 怎麼實作的？
本篇介紹一下一般各個作業系統的 C++ 編譯器是怎麼實作 std::mutex 的。

接下來我們來 trace llvm 的 libc++ 是怎麼實作 std::mutex 的。

1. <span style="font-weight:bold;">std::mutex::lock 的實作</span>
lock 轉換成內部的 __libcpp_mutex_lock

``` cpp
void mutex::lock()
{
    int ec = __libcpp_mutex_lock(&__m_);
    if (ec)
        __throw_system_error(ec, "mutex lock failed");
}
```

1-1. std::mutex::lock 在 unix 平台的實作內容
__libcpp_mutex_lock 在 unix 平台是呼叫 pthread_mutex_lock

``` cpp

int __libcpp_mutex_lock(__libcpp_mutex_t *__m)
{
  return pthread_mutex_lock(__m);
}
```

1-2. std::mutex::lock 在 windows 平台的實作內容
__libcpp_mutex_lock 在 windows 平台是呼叫 AcquireSRWLockExclusive

``` cpp
int __libcpp_mutex_lock(__libcpp_mutex_t *__m)
{
  AcquireSRWLockExclusive((PSRWLOCK)__m);
  return 0;
}
```

2. std::mutex::unlock 的實作
unlock 轉換成內部的 __libcpp_mutex_unlock

``` cpp
void mutex::unlock() _NOEXCEPT
{
    int ec = __libcpp_mutex_unlock(&__m_);
    (void)ec;
    _LIBCPP_ASSERT(ec == 0, "call to mutex::unlock failed");
}
```

2-1. std::mutex::unlock 在 unix 平台的實作內容
__libcpp_mutex_unlock 在 unix 平台是呼叫 pthread_mutex_unlock

``` cpp
int __libcpp_mutex_unlock(__libcpp_mutex_t *__m)
{
  return pthread_mutex_unlock(__m);
}
```

2-2. std::mutex::unlock 在 在 windows 平台的實作內容
__libcpp_mutex_unlock 在 windows 平台是呼叫 ReleaseSRWLockExclusive

``` cpp
int __libcpp_mutex_unlock(__libcpp_mutex_t *__m)
{
  ReleaseSRWLockExclusive((PSRWLOCK)__m);
  return 0;
}
```

總結
在一番 trace code 的結果後，std::mutex 在 llvm 的 libc++ 實作是這樣的，
在 unix 下
- std::mutex::lock 轉換成 pthread_mutex_lock
- std::mutex::unlock 轉換成 pthread_mutex_unlock
在 windows 下
- std::mutex::lock 轉換成 AcquireSRWLockExclusive
- std::mutex::unlock 轉換成 ReleaseSRWLockExclusive

