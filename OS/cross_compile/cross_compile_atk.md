git clone https://gitlab.gnome.org/GNOME/atk.git
cd atk
git checkout ATK_2_36_0

meson setup build-arm64 --cross-file /workspaces_data/repo/rk/aarch64.cross --prefix=/usr/aarch64-linux-gnu --debug -Dintrospection=false
ninja -C build-arm64
ninja -C build-arm64 install



如果你只是要 ATK runtime（給 GTK / OpenCV / GStreamer 用），99% 不需要 introspection。

直接關掉它：

meson setup build-arm64 \
  --cross-file /workspaces_data/repo/rk/aarch64.cross \
  --prefix=/usr/aarch64-linux-gnu \
  -Dintrospection=false

``` bash
root@ac4a29dc7b68:/workspaces_data/repo/rk/atk/atk# meson setup build-arm64 --cross-file /workspaces_data/repo/rk/aarch64.cross --prefix=/usr/aarch64-linux-gnu --debug
DEPRECATION: "pkgconfig" entry is deprecated and should be replaced by "pkg-config"
The Meson build system
Version: 1.10.0
Source dir: /workspaces_data/repo/rk/atk/atk
Build dir: /workspaces_data/repo/rk/atk/atk/build-arm64
Build type: cross build
Project name: atk
Project version: 2.36.0
C compiler for the host machine: /usr/bin/aarch64-linux-gnu-gcc (gcc 9.4.0 "aarch64-linux-gnu-gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
C linker for the host machine: /usr/bin/aarch64-linux-gnu-gcc ld.bfd 2.34
C compiler for the build machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
C linker for the build machine: cc ld.bfd 2.34
Build machine cpu family: x86_64
Build machine cpu: x86_64
Host machine cpu family: aarch64
Host machine cpu: armv8-a
Target machine cpu family: aarch64
Target machine cpu: armv8-a
Compiler for C supports arguments -fvisibility=hidden: YES 
Compiler for C supports link arguments -Wl,-Bsymbolic: YES 
Compiler for C supports link arguments -Wl,-z,relro: YES 
Compiler for C supports link arguments -Wl,-z,now: YES 
Checking for function "bind_textdomain_codeset" : YES 
meson.build:98: WARNING: Project targets '>= 0.46.0' but uses feature introduced in '0.54.0': fallback arg in dependency.
Found pkg-config: YES (/usr/bin/aarch64-linux-gnu-pkg-config) 0.29.1
Run-time dependency glib-2.0 found: YES 2.74.1
meson.build:100: WARNING: Project targets '>= 0.46.0' but uses feature introduced in '0.54.0': fallback arg in dependency.
Run-time dependency gobject-2.0 found: YES 2.74.1
Configuring config.h using configuration
Configuring atkversion.h using configuration
Found pkg-config: YES (/usr/bin/pkg-config) 0.29.1
Build-time dependency glib-2.0 found: YES 2.64.6
Program /usr/bin/glib-genmarshal found: YES (/usr/bin/glib-genmarshal)
Dependency glib-2.0 found: YES 2.64.6 (cached)
Program /usr/bin/glib-mkenums found: YES (/usr/bin/glib-mkenums)
Dependency glib-2.0 found: YES 2.64.6 (cached)
Program /usr/bin/glib-mkenums found: YES (/usr/bin/glib-mkenums)
Found CMake: NO
Run-time dependency gobject-introspection-1.0 found: NO (tried pkgconfig)

atk/meson.build:138:27: ERROR: Dependency "gobject-introspection-1.0" not found, tried pkgconfig

A full log can be found at /workspaces_data/repo/rk/atk/atk/build-arm64/meson-logs/meson-log.txt
```

這個錯誤其實很「Meson 風格」，而且你已經快解完這一關了。來把整個現象拆解一下。

你現在的狀態是典型的 cross build + GI（gobject-introspection）雙世界問題。

發生了什麼事（關鍵觀察）

注意這幾行，非常關鍵：

Run-time dependency glib-2.0 found: YES 2.74.1   (aarch64)
Build-time dependency glib-2.0 found: YES 2.64.6 (x86_64)


Meson 在 cross build 時刻意分裂世界：

host machine（aarch64）

連結用

走 /usr/bin/aarch64-linux-gnu-pkg-config

glib = 2.74.1 ✅

build machine（x86_64）

產生工具用（code generator）

走 /usr/bin/pkg-config

glib = 2.64.6 ✅

glib-genmarshal, glib-mkenums 都是 x86_64 ✅

目前為止都很正常。

為什麼 gobject-introspection 爆掉

錯在這行：

Run-time dependency gobject-introspection-1.0 found: NO (tried pkgconfig)


ATK 預設會啟用 GI，而 Meson 的規則是：

gobject-introspection
👉 build-time tool（x86_64） + runtime typelibs（target）

但你現在的環境是：

x86_64：

❌ 沒有 gobject-introspection-1.0.pc

aarch64 sysroot：

有沒有不重要，因為 Meson 在這一步要的是 build machine 的 GI

所以它直接死給你看。

三種正解路線（依你現在的目標）
✅ 路線 A：你根本不需要 GI（最常見、最乾淨）

如果你只是要 ATK runtime（給 GTK / OpenCV / GStreamer 用），99% 不需要 introspection。

直接關掉它：

meson setup build-arm64 \
  --cross-file /workspaces_data/repo/rk/aarch64.cross \
  --prefix=/usr/aarch64-linux-gnu \
  -Dintrospection=false


這是 embedded / cross build 的標準做法。

✔ 不需要 gobject-introspection
✔ 不會產生 .gir / .typelib
✔ ATK 功能完全正常

⚠ 路線 B：你真的要 GIR / Typelib（比較硬核）

那你要補齊 build machine 的 GI：

sudo apt install gobject-introspection libgirepository1.0-dev


然後確認：

pkg-config --exists gobject-introspection-1.0 && echo OK


注意是 x86_64 的 pkg-config，不是 aarch64 的。

但我要吐槽一句：
在 cross 編譯情境下搞 GI，維護成本很高，除非你明確需要 Python / JS binding，否則不值得。

🧪 路線 C：混合（進階玩法）

有些人會：

build machine：完整 GI

target：只裝 .typelib

然後手動指定：

-Dintrospection=enabled
-Dgi_cross_use_prebuilt_gir=true


但這已經是「我知道我在幹嘛」等級了，不建議現在走。

為什麼你之前裝了還是找不到（關鍵提醒）

你之前有提到：

apt install gobject-introspection libgirepository1.0-dev


但那是哪個系統？

如果是在 sysroot（/usr/aarch64-linux-gnu） → ❌ Meson 這裡用不到

Meson 現在要的是：

/usr/lib/pkgconfig/gobject-introspection-1.0.pc


（x86_64）

Meson 沒錯，是我們常常低估它有多嚴格 🤓