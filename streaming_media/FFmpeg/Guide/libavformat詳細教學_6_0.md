---
title: libavformat 詳細教學 (FFmpeg 6.0)
source: https://ffmpeg.org/doxygen/6.0/group__lavf__decoding.html
author:
published:
created: 2026-09-23
description: FFmpeg 6.0 libavformat 完整 API 教學，涵蓋所有 Demux/Mux 函式、核心資料結構與完整流程說明。
categorization: streaming_media/FFmpeg
tags:
  - FFmpeg
  - libavformat
  - demuxer
  - muxer
  - AVFormatContext
  - AVPacket
  - AVStream
  - FFmpeg-6_0
---

## 概述

libavformat 提供通用的 mux/demux 框架，支援多種容器格式的 multiplexing 與 demultiplexing，並支援多種 I/O 協定存取媒體資源。

相關筆記：[[FFmpeg模組種類總覽]]、[[demuxer-module]]

---

## 核心資料結構

### AVFormatContext
整個 Format I/O 的核心 context，定義於 `avformat.h:1104`。所有欄位可透過 AVOptions (`av_opt*`) 存取，對應 command line 參數名稱。不得使用 `sizeof(AVFormatContext)`，應使用 `avformat_alloc_context()` 建立實例。

```c
AVFormatContext {
    const AVInputFormat  *iformat;       // 輸入格式（demux 時）
    const AVOutputFormat *oformat;       // 輸出格式（mux 時）
    AVIOContext          *pb;            // I/O context（可自訂）
    unsigned int          nb_streams;   // 串流數量（唯讀，由 avformat_new_stream 設定）
    AVStream            **streams;      // 所有串流陣列
    char                 *url;          // 輸入/輸出 URL
    int64_t               duration;     // 總時長（AV_TIME_BASE 單位）
    int64_t               bit_rate;     // 整體位元率
    AVDictionary         *metadata;     // 容器 metadata
    int64_t               max_interleave_delta; // av_interleaved_write_frame 最大緩衝時長
}
```

### AVStream
每個串流（video/audio/subtitle）對應一個 `AVStream`。

```c
AVStream {
    int                  index;         // 在 AVFormatContext.streams 中的索引
    AVCodecParameters   *codecpar;      // codec 參數（取代舊的 codec）
    AVRational           time_base;     // 時間基準（如 1/90000）
    int64_t              duration;      // 串流時長（time_base 單位）
    int64_t              nb_frames;     // 已知總幀數（可能為 0）
    AVRational           avg_frame_rate; // 平均幀率
    AVDiscard            discard;       // 設定丟棄哪些 packets
    AVDictionary        *metadata;      // per-stream metadata
}
```

### AVPacket
儲存壓縮的 encoded 資料，屬於某一條 AVStream。

```c
AVPacket {
    AVBufferRef *buf;          // reference-counted buffer
    int64_t      pts;          // Presentation timestamp（AVStream.time_base 單位）
    int64_t      dts;          // Decompression timestamp
    uint8_t     *data;         // 資料指標
    int          size;         // 資料大小（bytes）
    int          stream_index; // 所屬串流索引
    int          flags;        // AV_PKT_FLAG_KEY 等旗標
    int64_t      duration;     // 時長（AVStream.time_base 單位）
    int64_t      pos;          // 在檔案中的位置（-1 表示未知）
}
```

### AVOutputFormat
Mux 時使用，描述輸出容器格式的能力與行為。

```c
AVOutputFormat {
    const char *name;          // 格式短名稱（如 "mp4"、"matroska"）
    const char *long_name;     // 格式全名
    const char *mime_type;
    const char *extensions;    // 副檔名（逗號分隔，如 "mp4,m4v"）
    enum AVCodecID video_codec;  // 預設 video codec
    enum AVCodecID audio_codec;  // 預設 audio codec
    int flags;                   // AVFMT_* 旗標
}
```

### AVProgram
廣播格式（MPEG-TS、DVB）中的 program 概念，包含一組相關串流。

```c
AVProgram {
    int           id;           // program ID
    int           flags;
    enum AVDiscard discard;
    unsigned int *stream_index; // 屬於此 program 的串流索引陣列
    unsigned int  nb_stream_indexes;
    AVDictionary *metadata;
    int           program_num;
    int           pmt_pid;
    int           pcr_pid;
}
```

---

## Demux 流程（解封裝）

### 完整呼叫流程

```
avformat_alloc_context()           ← 可選，需自訂 pb 時使用
         ↓
avformat_open_input()              ← 開啟輸入、讀取 header
         ↓
avformat_find_stream_info()        ← 讀若干 packet 補齊串流資訊
         ↓
av_find_best_stream()              ← 找到最佳 video/audio 串流索引
    或
av_find_program_from_stream()      ← 廣播格式：先找 program，再選串流
         ↓
av_read_frame() 迴圈               ← 逐 packet 讀取
         ↓
av_packet_unref()                  ← 釋放 packet（迴圈內）
         ↓
[av_seek_frame() / avformat_seek_file()]  ← 可選，Seek 操作
         ↓
[avformat_flush()]                 ← 可選，清除內部緩衝
         ↓
avformat_close_input()             ← 關閉、釋放所有資源
```

---

## Demux API 函式詳解

### 1. `avformat_alloc_context()`

```c
AVFormatContext *avformat_alloc_context(void);
// 回傳：已分配的 AVFormatContext，失敗回傳 NULL
```

手動分配 AVFormatContext，用於需要在 `avformat_open_input()` 之前進行客製化設定的場景，例如設定自訂 `AVIOContext`（`pb` 欄位）。若無特殊需求，可直接傳 NULL 給 `avformat_open_input()`，讓它自動分配。

---

### 2. `avformat_open_input()`

```c
int avformat_open_input(
    AVFormatContext **ps,           // 輸出：AVFormatContext 指標（傳 NULL 自動分配）
    const char      *url,           // 輸入 URL 或檔案路徑
    const AVInputFormat *fmt,       // 強制指定格式（NULL = 自動偵測）
    AVDictionary    **options       // demuxer 私有選項（回傳時包含未識別選項）
);
// 回傳：0 成功，負值 AVERROR
```

嘗試分配 AVFormatContext、開啟指定 URL（自動偵測格式）並讀取 header。codec 不會在此時開啟。若使用者提供的 AVFormatContext 在失敗時會被釋放。

**基本用法：**

```c
AVFormatContext *fmt_ctx = NULL;
int ret = avformat_open_input(&fmt_ctx, "file:input.mp4", NULL, NULL);
if (ret < 0) {
    char errbuf[128];
    av_strerror(ret, errbuf, sizeof(errbuf));
    fprintf(stderr, "開啟失敗: %s\n", errbuf);
    return ret;
}
```

**傳入 demuxer 私有選項：**

```c
AVDictionary *opts = NULL;
av_dict_set(&opts, "video_size", "640x480", 0);
av_dict_set(&opts, "pixel_format", "rgb24", 0);

if (avformat_open_input(&fmt_ctx, url, NULL, &opts) < 0)
    abort();

// 檢查未被識別的選項（應在呼叫後檢查）
AVDictionaryEntry *e = NULL;
while ((e = av_dict_get(opts, "", e, AV_DICT_IGNORE_SUFFIX)))
    fprintf(stderr, "未識別選項: %s\n", e->key);

av_dict_free(&opts);
```

> **重要**：由於格式在 `avformat_open_input()` 返回前通常無法確定，demuxer 私有選項無法在 preallocated context 上直接設定，必須透過 `AVDictionary` 傳入。

**自訂 I/O（Custom AVIOContext）：**

```c
AVFormatContext *fmt_ctx = avformat_alloc_context();
fmt_ctx->pb = avio_alloc_context(buf, buf_size, 0, opaque,
                                  read_cb, NULL, seek_cb);
// url 傳 NULL，因為資料來源是自訂 pb
avformat_open_input(&fmt_ctx, NULL, NULL, NULL);
```

---

### 3. `avformat_find_stream_info()`

```c
int avformat_find_stream_info(
    AVFormatContext *ic,        // 已開啟的 context
    AVDictionary   **options    // 各串流 codec 選項陣列（可 NULL）
);
// 回傳：>=0 成功，AVERROR_xxx 失敗
```

對於沒有 header 或 header 資訊不足的格式（如裸 MPEG），此函式讀取並解碼若干 packet 以補齊 stream 資訊（包含真實 framerate）。邏輯讀取位置不受影響，讀取的 packet 會被緩衝供後續處理。

> **注意**：此函式不保證開啟所有 codec，回傳時 options 非空屬正常行為。

```c
ret = avformat_find_stream_info(fmt_ctx, NULL);
if (ret < 0) {
    fprintf(stderr, "無法取得串流資訊\n");
    avformat_close_input(&fmt_ctx);
    return ret;
}

// 印出所有串流資訊（除錯用）
av_dump_format(fmt_ctx, 0, url, 0);
```

---

### 4. `av_find_best_stream()`

```c
int av_find_best_stream(
    AVFormatContext     *ic,
    enum AVMediaType     type,              // AVMEDIA_TYPE_VIDEO / AUDIO / SUBTITLE
    int                  wanted_stream_nb,  // 使用者指定串流號，-1 = 自動
    int                  related_stream,    // 關聯串流索引（同 program 優先），-1 = 無
    const AVCodec      **decoder_ret,       // 輸出：對應 decoder（可 NULL）
    int                  flags              // 目前無定義，傳 0
);
// 回傳：非負串流索引
//       AVERROR_STREAM_NOT_FOUND：找不到指定類型串流
//       AVERROR_DECODER_NOT_FOUND：有串流但找不到 decoder
```

以啟發式規則找到「最佳」串流。若 `decoder_ret` 非 NULL，會找出對應的預設 decoder；找不到 decoder 的串流會被忽略。成功且 `decoder_ret` 非 NULL 時，`*decoder_ret` 保證指向有效的 `AVCodec`。

```c
const AVCodec *video_codec = NULL;
int video_stream_idx = av_find_best_stream(fmt_ctx,
                                            AVMEDIA_TYPE_VIDEO,
                                            -1, -1,
                                            &video_codec, 0);
if (video_stream_idx < 0) {
    fprintf(stderr, "找不到 video 串流\n");
    return video_stream_idx;
}

AVStream *video_stream = fmt_ctx->streams[video_stream_idx];
```

---

### 5. `av_find_program_from_stream()`

```c
AVProgram *av_find_program_from_stream(
    AVFormatContext *ic,     // media file handle
    AVProgram       *last,   // 上一次找到的 program，NULL 表示從頭搜尋
    int              s       // stream index
);
// 回傳：下一個包含該 stream 的 AVProgram，找不到則為 NULL
```

找出包含指定串流索引的 AVProgram。主要用於 MPEG-TS、ATSC、DVB 等廣播格式，其中一個 program 包含一組相關的 video/audio/subtitle stream（例如同一個頻道的影音字幕）。搭配 `av_find_best_stream()` 的 `related_stream` 參數，可優先在同 program 內選取串流。

```c
// 找出 stream 0 所屬的所有 program
AVProgram *prog = NULL;
while ((prog = av_find_program_from_stream(fmt_ctx, prog, 0)) != NULL) {
    printf("Program id=%d, 包含 %d 條串流\n",
           prog->id, prog->nb_stream_indexes);
    for (unsigned int i = 0; i < prog->nb_stream_indexes; i++)
        printf("  stream #%u\n", prog->stream_index[i]);
}

// 在同一 program 內找最佳 audio 串流
// 先找 video stream，再以 video_stream_idx 為 related_stream 找 audio
int audio_stream_idx = av_find_best_stream(fmt_ctx,
                                            AVMEDIA_TYPE_AUDIO,
                                            -1, video_stream_idx,
                                            NULL, 0);
```

---

### 6. `av_program_add_stream_index()`

```c
void av_program_add_stream_index(
    AVFormatContext *ac,
    int              progid,    // program ID
    unsigned int     idx        // stream index
);
```

將指定串流索引加入對應 program ID 的 AVProgram。若 program 不存在則自動建立。主要由 demuxer 內部（如 mpegts demuxer）呼叫，一般使用者直接操作較少。

---

### 7. `av_read_frame()`

```c
int av_read_frame(
    AVFormatContext *s,   // 已開啟的 context
    AVPacket        *pkt  // 輸出：讀取到的 packet（需先 av_packet_alloc）
);
// 回傳：0 成功，< 0 錯誤或 EOF（AVERROR_EOF）
```

回傳下一個 stream 的 packet，直接對應檔案儲存的資料，不驗證資料有效性。回傳的 packet 永遠是 reference-counted（`pkt->buf` 已設定），可無限期持有，使用完畢必須呼叫 `av_packet_unref()` 釋放。

- **Video**：每次恰好一個 frame
- **Audio（固定大小，如 PCM/ADPCM）**：可能包含多個 frame
- **Audio（可變大小，如 MPEG audio）**：一個 frame
- `pts`/`dts`/`duration` 皆以 `AVStream.time_base` 為單位，若格式無法提供則可能為 `AV_NOPTS_VALUE`/0

> **注意**：B-frame 影片中 `pkt->pts` 可能為 `AV_NOPTS_VALUE`，此時應依賴 `pkt->dts`。

```c
AVPacket *pkt = av_packet_alloc();

while (av_read_frame(fmt_ctx, pkt) >= 0) {
    if (pkt->stream_index == video_stream_idx) {
        // 送入 libavcodec decoder
        // avcodec_send_packet(dec_ctx, pkt);
    } else if (pkt->stream_index == audio_stream_idx) {
        // 處理 audio
    }
    av_packet_unref(pkt);  // 每次迴圈都必須釋放！
}

av_packet_free(&pkt);
```

---

### 8. `av_seek_frame()`

```c
int av_seek_frame(
    AVFormatContext *s,
    int              stream_index,  // -1 = 使用預設串流（單位自動換算為 AV_TIME_BASE）
    int64_t          timestamp,     // 目標 timestamp（AVStream.time_base 單位）
    int              flags          // AVSEEK_FLAG_* 旗標
);
// 回傳：>= 0 成功
```

Seek 到指定 timestamp 的 keyframe。

`flags` 常用值：

| 旗標 | 值 | 說明 |
|---|---|---|
| `AVSEEK_FLAG_BACKWARD` | 1 | Seek 到 timestamp 之前最近的 keyframe |
| `AVSEEK_FLAG_BYTE` | 2 | timestamp 以位元組偏移計算 |
| `AVSEEK_FLAG_ANY` | 4 | 允許 seek 到非 keyframe |
| `AVSEEK_FLAG_FRAME` | 8 | timestamp 以幀數計算 |

```c
// Seek 到 30 秒（stream_index=-1 時使用 AV_TIME_BASE）
int64_t target = 30LL * AV_TIME_BASE;
av_seek_frame(fmt_ctx, -1, target, AVSEEK_FLAG_BACKWARD);
```

---

### 9. `avformat_seek_file()`

```c
int avformat_seek_file(
    AVFormatContext *s,
    int              stream_index,  // 時間基準參考串流，-1 = AV_TIME_BASE
    int64_t          min_ts,        // 最小可接受 timestamp
    int64_t          ts,            // 目標 timestamp
    int64_t          max_ts,        // 最大可接受 timestamp
    int              flags
);
// 回傳：>= 0 成功
```

精確 Seek API（仍在建設中）。確保所有 active 串流（`AVStream.discard < AVDISCARD_ALL`）都能從該點正常播放。支援 `AVSEEK_FLAG_BYTE`、`AVSEEK_FLAG_FRAME`、`AVSEEK_FLAG_ANY`。

```c
// Seek 到 30 秒，允許 ±1 秒誤差
int64_t ts = 30LL * AV_TIME_BASE;
avformat_seek_file(fmt_ctx, -1,
                   ts - AV_TIME_BASE,  // min_ts
                   ts,                  // 目標
                   ts + AV_TIME_BASE,  // max_ts
                   0);
```

---

### 10. `avformat_flush()`

```c
int avformat_flush(AVFormatContext *s);
// 回傳：>= 0 成功
```

丟棄所有內部緩衝資料。適用於處理 byte stream 中不連續點（discontinuity），通常搭配 live stream 重新同步。支援可重新同步的格式（MPEG-TS、NUT、Ogg 等）。不會改變串流集合、時長與 codec 參數，也不 flush `AVIOContext`（若需要請先呼叫 `avio_flush(s->pb)`）。若需完整 reset 建議重新開啟 AVFormatContext。

---

### 11. `av_read_play()` / `av_read_pause()`

```c
int av_read_play(AVFormatContext *s);   // 開始播放網路串流（如 RTSP）
int av_read_pause(AVFormatContext *s);  // 暫停網路串流，以 av_read_play() 恢復
```

僅對網路串流有意義（RTSP 等），本地檔案呼叫無效果。

---

### 12. `avformat_close_input()`

```c
void avformat_close_input(AVFormatContext **s);
// 釋放所有相關記憶體，並將 *s 設為 NULL
```

關閉已開啟的輸入 AVFormatContext，釋放一切相關資源。

---

### 13. 格式探測（Format Probing）

```c
// 根據短名稱找 AVInputFormat（如 "mp4"、"rtsp"）
const AVInputFormat *av_find_input_format(const char *short_name);
// 例：av_find_input_format("mp4")、av_find_input_format("h264")
// 定義於 format.c:118
// 已知格式名稱時直接查，不需要讀資料。
// 根據 AVProbeData 猜測格式（低階，一般不直接使用）
typedef struct AVProbeData {
    const char *filename;  // 檔名（可 NULL）
    unsigned char *buf;    // 待探測的原始資料（至少 AVPROBE_PADDING_SIZE 後綴為 0）
    int buf_size;          // buf 大小（不含後綴）
    const char *mime_type; // MIME type（可 NULL）
} AVProbeData;

const AVInputFormat *av_probe_input_format(
const AVProbeData *pd, // 待探測資料
int is_opened // 檔案是否已開啟（決定探測 AVFMT_NOFILE 格式與否）
);
// 定義於 format.c:219
// 內部呼叫 av_probe_input_format2()，score_max 門檻固定
// 最簡單的探測介面，不回傳分數，只回傳猜測到的格式。

// 同上，帶 score 閾值
const AVInputFormat *av_probe_input_format2(const AVProbeData *pd,
                                             int is_opened,
                                             int *score_max // 輸入：最低接受分數；輸出：實際偵測分數
                                             );
// 定義於 format.c:207
// 若偵測分數 ≤ `AVPROBE_SCORE_MAX / 4`，官方建議用更大的 probe buffer 重試。`score_max` 兼做輸入門檻與輸出結果，呼叫後讀回即可得知可信度。
// 同上，回傳最高 score
const AVInputFormat *av_probe_input_format3(const AVProbeData *pd,
                                             int is_opened,
                                             int *score_ret // 輸出：最佳偵測分數
                                             );
// 定義於 format.c:128

// 三者中最底層。`av_probe_input_format2()` 和 `av_probe_input_format()` 都是包它的 wrapper。內部迭代所有已知 demuxer，取最高分者回傳。
// 三者關係：
// av_probe_input_format()
//└→ av_probe_input_format2()
//└→ av_probe_input_format3() ← 真正掃所有 demuxer


// 從 bytestream 探測格式，分數不夠時自動增大 buffer 重試
// 回傳實際分數（成功為正值）
int av_probe_input_buffer2(AVIOContext   *pb,
                            const AVInputFormat **fmt,
                            const char    *url,
                            void          *logctx,
                            unsigned int   offset,
                            unsigned int   max_probe_size);
// 同上，成功回傳 0
int av_probe_input_buffer(AVIOContext *pb, const AVInputFormat **fmt,
                           const char *url, void *logctx,
                           unsigned int offset, unsigned int max_probe_size);
                           // 定義於 format.c:315
                           // `av_probe_input_buffer2()` 的探測流程：每次探測分數過低時增大 probe buffer 重試，直到達到 `max_probe_size`（0 = 使用預設值）或找到格式，取分數最高的格式回傳。
                           // 適用場景：已有 `AVIOContext`（如自訂 I/O、記憶體緩衝），需要在 `avformat_open_input()` 之前手動探測格式。分數不足時會自動擴大 probe buffer 重試，直到 `max_probe_size`（0 = 預設值）。
```

**典型使用情境（記憶體串流）：**
```c
AVIOContext *avio_ctx = NULL;
const AVInputFormat *fmt = NULL;

// 從自訂 buffer 建立 AVIOContext
avio_ctx = avio_alloc_context(buffer, buf_size, 0, NULL,
                               my_read_cb, NULL, my_seek_cb);

// 探測格式
int score = av_probe_input_buffer2(avio_ctx, &fmt, "", NULL, 0, 0);
if (score < 0) {
    fprintf(stderr, "探測格式失敗\n");
    goto end;
}
printf("偵測到格式: %s (score=%d)\n", fmt->name, score);

// 再用探測到的格式開啟
AVFormatContext *fmt_ctx = avformat_alloc_context();
fmt_ctx->pb = avio_ctx;
avformat_open_input(&fmt_ctx, "", fmt, NULL);
```

---

**探測函式選擇指南：**

| 情境                             | 建議函式                                                    |
| ------------------------------ | ------------------------------------------------------- |
| 已知格式名稱（如 `"mp4"`）              | `av_find_input_format()`                                |
| 有原始 buffer，只要結果                | `av_probe_input_format()`                               |
| 有原始 buffer，需要知道可信度             | `av_probe_input_format2()` / `av_probe_input_format3()` |
| 有 `AVIOContext`（自訂 I/O / 網路串流） | `av_probe_input_buffer()`                               |
| 通常情況（從 URL/檔案開啟）               | 直接 `avformat_open_input()` 即可（內部自動探測）                   |

---

## Mux 流程（封裝輸出）

### 完整呼叫流程

```
avformat_alloc_output_context2()   ← 建立輸出 context，指定格式
         ↓
avformat_new_stream()              ← 新增輸出串流（video/audio）
填入 AVStream.codecpar             ← 設定 codec 參數、time_base
         ↓
avio_open2() / 自訂 pb             ← 開啟輸出 I/O（AVFMT_NOFILE 格式不需要）
         ↓
[avformat_init_output()]           ← 可選：提前初始化，不寫 header
         ↓
avformat_write_header()            ← 初始化 muxer，寫入 container header
         ↓
av_write_frame()                   ← 寫入 packet（呼叫者負責交錯）
   或
av_interleaved_write_frame()       ← 寫入 packet（libavformat 自動交錯）
   注意：同一個 context 只能選一種，不可混用
         ↓
av_write_trailer()                 ← 寫入結尾、finalize 容器
         ↓
avio_closep(&fmt_ctx->pb)          ← 關閉 I/O（AVFMT_NOFILE 不需要）
         ↓
avformat_free_context()            ← 釋放 context
```

---

## Mux API 函式詳解

### 14. `avformat_alloc_output_context2()`

```c
int avformat_alloc_output_context2(
    AVFormatContext **ctx,          // 輸出：分配的 context
    const AVOutputFormat *oformat,  // 明確指定輸出格式（可 NULL）
    const char *format_name,        // 格式短名稱（可 NULL）
    const char *filename            // 輸出檔名（用於自動判斷格式）
);
// 回傳：0 成功，負值 AVERROR
```

```c
AVFormatContext *out_ctx = NULL;
// 根據副檔名自動判斷格式
avformat_alloc_output_context2(&out_ctx, NULL, NULL, "output.mp4");
if (!out_ctx) { fprintf(stderr, "無法判斷輸出格式\n"); return -1; }
```

---

### 15. `avformat_new_stream()`

```c
AVStream *avformat_new_stream(
    AVFormatContext *s,
    const AVCodec   *c    // 可 NULL（6.0 建議傳 NULL，透過 codecpar 設定）
);
// 回傳：新建立的 AVStream，失敗回傳 NULL
```

新增一條輸出串流。

```c
AVStream *out_stream = avformat_new_stream(out_ctx, NULL);
if (!out_stream) return -1;

// 複製 codec 參數（remux 場景）
avcodec_parameters_copy(out_stream->codecpar,
                         in_fmt_ctx->streams[video_idx]->codecpar);
out_stream->codecpar->codec_tag = 0;  // 讓 muxer 自行決定

// 設定 time_base（muxer 可能在 write_header 後修改）
out_stream->time_base = in_fmt_ctx->streams[video_idx]->time_base;
```

> **注意**：remux 時建議手動只填寫相關的 `AVCodecParameters` 欄位，而非整個 `avcodec_parameters_copy()`，因為不保證所有欄位對輸入和輸出都同樣有效。

---

### 16. `avformat_init_output()`

```c
int avformat_init_output(
    AVFormatContext  *s,
    AVDictionary    **options
);
// 回傳：AVSTREAM_INIT_IN_WRITE_HEADER（0）或 AVSTREAM_INIT_IN_INIT_OUTPUT（1），負值失敗
```

可選步驟，在 `avformat_write_header()` 之前提前初始化 codec，但不寫入 header。適用於需要在 header 寫入前就知道確切 stream 參數的情境（例如 DASH、HLS 等分段格式）。呼叫此函式後，**不要**再把相同的 options 傳給 `avformat_write_header()`。

回傳值含義：
- `AVSTREAM_INIT_IN_WRITE_HEADER`（0）：codec 尚未完整初始化，還需 `avformat_write_header()` 完成
- `AVSTREAM_INIT_IN_INIT_OUTPUT`（1）：codec 已完整初始化

---

### 17. `avformat_write_header()`

```c
int avformat_write_header(
    AVFormatContext  *s,
    AVDictionary    **options   // muxer 私有選項（回傳時包含未識別選項）
);
// 回傳：AVSTREAM_INIT_IN_WRITE_HEADER（0）或 AVSTREAM_INIT_IN_INIT_OUTPUT（1），負值失敗
```

初始化 muxer 內部並寫入 container header。無論 muxer 是否實際寫入 I/O，此函式都必須呼叫。

```c
ret = avformat_write_header(out_ctx, NULL);
if (ret < 0) {
    fprintf(stderr, "寫入 header 失敗\n");
    return ret;
}
// 注意：此後 AVStream.time_base 可能已被 muxer 修改
// 後續寫入 packet 時需要用 av_rescale_q 換算 timestamp
```

---

### 18. `av_write_frame()`

```c
int av_write_frame(
    AVFormatContext *s,
    AVPacket        *pkt   // pkt=NULL 可立即 flush muxer 內部緩衝
);
// 回傳：< 0 錯誤，= 0 成功，= 1 已 flush 且無更多資料
```

直接將 packet 送給 muxer，不做任何緩衝或重新排序。呼叫者需自行確保交錯順序（DTS 遞增）。不取得 packet 的擁有權（部分 muxer 可能建立內部副本）。

- Packet 的 `stream_index` 必須設定為對應串流索引
- `pts`/`dts` 必須是 `AVStream.time_base` 單位的正確值（格式有 `AVFMT_NOTIMESTAMPS` 旗標時可為 `AV_NOPTS_VALUE`）
- 同一串流後續 packet 的 DTS 必須嚴格遞增（有 `AVFMT_TS_NONSTRICT` 旗標時只需非遞減）

---

### 19. `av_interleaved_write_frame()`

```c
int av_interleaved_write_frame(
    AVFormatContext *s,
    AVPacket        *pkt   // pkt=NULL 可 flush 交錯佇列
);
// 回傳：0 成功，負值 AVERROR
```

將 packet 送給 muxer，由 libavformat 自動緩衝並以正確 DTS 順序交錯輸出。取得 reference-counted packet 的擁有權（呼叫後 pkt 變空白）；非 reference-counted packet 會被複製。可透過 `AVFormatContext.max_interleave_delta` 控制最大緩衝時長。比 `av_write_frame()` 多一層緩衝，對 mp4 VFR 分段模式等場景有額外優化。

> **重要**：`av_write_frame()` 與 `av_interleaved_write_frame()` 對同一個 context **不可混用**。

```c
// Remux 典型寫法
AVPacket *pkt = av_packet_alloc();
while (av_read_frame(in_fmt_ctx, pkt) >= 0) {
    AVStream *in_stream  = in_fmt_ctx->streams[pkt->stream_index];
    AVStream *out_stream = out_ctx->streams[pkt->stream_index];

    // 換算 timestamp 到輸出串流的 time_base
    av_packet_rescale_ts(pkt, in_stream->time_base, out_stream->time_base);
    pkt->pos = -1;

    ret = av_interleaved_write_frame(out_ctx, pkt);
    // 注意：呼叫後 pkt 已被清空，不需再 unref
    if (ret < 0) break;
}
av_packet_free(&pkt);
```

---

### 20. `av_write_uncoded_frame()`

```c
int av_write_uncoded_frame(
    AVFormatContext *s,
    int              stream_index,
    AVFrame         *frame
);
```

直接寫入未編碼的 AVFrame（不經過 AVPacket），主要用於支援 raw video/PCM 的裝置或特殊 muxer。呼叫者需自行確保交錯順序。呼叫後失去 frame 的擁有權。

---

### 21. `av_interleaved_write_uncoded_frame()`

```c
int av_interleaved_write_uncoded_frame(
    AVFormatContext *s,
    int              stream_index,
    AVFrame         *frame
);
// 回傳：>= 0 成功
```

同 `av_write_uncoded_frame()`，但由 libavformat 自動處理交錯。支援的 muxer 可用 `av_write_uncoded_frame_query()` 查詢。

---

### 22. `av_write_uncoded_frame_query()`

```c
int av_write_uncoded_frame_query(
    AVFormatContext *s,
    int              stream_index
);
// 回傳：>= 0 表示支援，< 0 表示不支援
```

查詢指定 muxer 與串流是否支援 uncoded frame 寫入。使用 `av_write_uncoded_frame()` 或 `av_interleaved_write_uncoded_frame()` 前應先查詢。

---

### 23. `av_write_trailer()`

```c
int av_write_trailer(AVFormatContext *s);
// 回傳：0 成功，AVERROR_xxx 失敗
// 必須在 avformat_write_header() 成功後才能呼叫
```

將所有內部緩衝 packet flush 輸出，寫入 container 結尾資訊（如 MP4 的 moov atom），釋放 muxer 私有資料。呼叫後應關閉 I/O context 並釋放 AVFormatContext。

```c
av_write_trailer(out_ctx);

// 關閉輸出 I/O（非 AVFMT_NOFILE 格式）
if (!(out_ctx->oformat->flags & AVFMT_NOFILE))
    avio_closep(&out_ctx->pb);

avformat_free_context(out_ctx);
```

---

### 24. `av_guess_format()`

```c
const AVOutputFormat *av_guess_format(
    const char *short_name,   // 格式短名稱（可 NULL）
    const char *filename,     // 檔名（根據副檔名判斷，可 NULL）
    const char *mime_type     // MIME type（可 NULL）
);
// 回傳：最匹配的 AVOutputFormat，無匹配回傳 NULL
```

根據短名稱、檔名副檔名或 MIME type 找出對應的輸出格式。`avformat_alloc_output_context2()` 內部也是呼叫此函式。

```c
const AVOutputFormat *fmt = av_guess_format(NULL, "output.mkv", NULL);
if (fmt) printf("猜測格式: %s\n", fmt->name);  // "matroska"
```

---

### 25. `av_guess_codec()`

```c
enum AVCodecID av_guess_codec(
    const AVOutputFormat *fmt,
    const char *short_name,   // 可 NULL
    const char *filename,     // 可 NULL
    const char *mime_type,    // 可 NULL
    enum AVMediaType type     // AVMEDIA_TYPE_VIDEO / AUDIO
);
// 回傳：最適合此格式的 AVCodecID
```

根據 muxer 與檔名猜測最適合的 codec ID。例如 mp4 格式 + video → `AV_CODEC_ID_H264`。

---

### 26. `av_get_output_timestamp()`

```c
int av_get_output_timestamp(
    struct AVFormatContext *s,
    int     stream,       // 串流索引
    int64_t *dts,         // 輸出：目前最後一個 packet 的 DTS（stream time_base 單位）
    int64_t *wall         // 輸出：該 packet 輸出的絕對時間（微秒）
);
// 回傳：0 成功，AVERROR(ENOSYS) 不支援
```

取得目前輸出資料的時間資訊，主要用於有內部緩衝或即時工作的裝置（如 V4L2、音效卡）。部分格式或裝置可能無法原子性地同時量測 DTS 與 wall time。

---

## Macro 常數

```c
// avformat_write_header() / avformat_init_output() 回傳值
#define AVSTREAM_INIT_IN_WRITE_HEADER  0  // stream 參數在 write_header 中初始化
#define AVSTREAM_INIT_IN_INIT_OUTPUT   1  // stream 參數在 init_output 中初始化
```

---

## 時間戳記換算

libavformat 全程使用有理數時間基準（AVRational）。

```c
// AVStream.time_base 的 timestamp → 秒
double ts_seconds = (double)pkt->pts * av_q2d(stream->time_base);

// 跨串流換算（如 video time_base → audio time_base）
int64_t converted = av_rescale_q(pkt->pts,
                                  video_stream->time_base,
                                  audio_stream->time_base);

// 整包換算（remux 用）
av_packet_rescale_ts(pkt, in_stream->time_base, out_stream->time_base);

// timestamp 轉字串（除錯用）
char ts_str[AV_TS_MAX_STRING_SIZE];
av_ts_make_time_string(ts_str, pkt->pts, &stream->time_base);
```

---

## 常見錯誤處理

```c
char errbuf[256];
av_strerror(ret, errbuf, sizeof(errbuf));
fprintf(stderr, "錯誤 %d: %s\n", ret, errbuf);

// 常見 AVERROR
// AVERROR(ENOMEM)            記憶體不足
// AVERROR(EINVAL)            參數無效
// AVERROR(EIO)               I/O 錯誤
// AVERROR_EOF                檔案結尾
// AVERROR_INVALIDDATA        資料損毀
// AVERROR_STREAM_NOT_FOUND   找不到串流
// AVERROR_DECODER_NOT_FOUND  找不到 decoder
// AVERROR(ENOSYS)            功能不支援
```

---

## 完整 Demux 範例（C）

```c
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

int main(int argc, char *argv[])
{
    const char *url = argv[1];
    AVFormatContext *fmt_ctx = NULL;
    AVPacket *pkt = NULL;
    int ret, video_idx;

    /* 1. 開啟輸入 */
    ret = avformat_open_input(&fmt_ctx, url, NULL, NULL);
    if (ret < 0) goto end;

    /* 2. 補齊串流資訊 */
    ret = avformat_find_stream_info(fmt_ctx, NULL);
    if (ret < 0) goto end;

    av_dump_format(fmt_ctx, 0, url, 0);

    /* 3. 找最佳 video 串流 */
    video_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO,
                                    -1, -1, NULL, 0);
    if (video_idx < 0) { ret = video_idx; goto end; }

    /* 4. 讀取 packet 迴圈 */
    pkt = av_packet_alloc();
    while ((ret = av_read_frame(fmt_ctx, pkt)) >= 0) {
        if (pkt->stream_index == video_idx) {
            AVStream *st = fmt_ctx->streams[video_idx];
            int64_t pts_ms = av_rescale_q(pkt->pts, st->time_base,
                                           (AVRational){1, 1000});
            printf("video pkt: size=%d pts_ms=%" PRId64 " key=%d\n",
                   pkt->size, pts_ms,
                   !!(pkt->flags & AV_PKT_FLAG_KEY));
        }
        av_packet_unref(pkt);
    }

end:
    av_packet_free(&pkt);
    avformat_close_input(&fmt_ctx);
    return (ret < 0 && ret != AVERROR_EOF) ? 1 : 0;
}
```

## 完整 Remux 範例（C）

```c
#include <libavformat/avformat.h>

int main(int argc, char *argv[])
{
    AVFormatContext *in_ctx = NULL, *out_ctx = NULL;
    AVPacket *pkt = av_packet_alloc();
    int ret;

    /* 1. 開啟輸入 */
    if ((ret = avformat_open_input(&in_ctx, argv[1], NULL, NULL)) < 0) goto end;
    if ((ret = avformat_find_stream_info(in_ctx, NULL)) < 0) goto end;

    /* 2. 建立輸出 context */
    if ((ret = avformat_alloc_output_context2(&out_ctx, NULL, NULL, argv[2])) < 0)
        goto end;

    /* 3. 複製所有串流 */
    for (unsigned int i = 0; i < in_ctx->nb_streams; i++) {
        AVStream *in_st  = in_ctx->streams[i];
        AVStream *out_st = avformat_new_stream(out_ctx, NULL);
        if (!out_st) { ret = AVERROR(ENOMEM); goto end; }
        avcodec_parameters_copy(out_st->codecpar, in_st->codecpar);
        out_st->codecpar->codec_tag = 0;
    }

    /* 4. 開啟輸出 I/O */
    if (!(out_ctx->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&out_ctx->pb, argv[2], AVIO_FLAG_WRITE)) < 0)
            goto end;
    }

    /* 5. 寫 header */
    if ((ret = avformat_write_header(out_ctx, NULL)) < 0) goto end;

    /* 6. 轉送所有 packet */
    while (av_read_frame(in_ctx, pkt) >= 0) {
        AVStream *in_st  = in_ctx->streams[pkt->stream_index];
        AVStream *out_st = out_ctx->streams[pkt->stream_index];
        av_packet_rescale_ts(pkt, in_st->time_base, out_st->time_base);
        pkt->pos = -1;
        av_interleaved_write_frame(out_ctx, pkt);
        // pkt 已被清空，不需 unref
    }

    /* 7. 寫 trailer，關閉資源 */
    av_write_trailer(out_ctx);

end:
    av_packet_free(&pkt);
    avformat_close_input(&in_ctx);
    if (out_ctx && !(out_ctx->oformat->flags & AVFMT_NOFILE))
        avio_closep(&out_ctx->pb);
    avformat_free_context(out_ctx);
    return ret < 0 ? 1 : 0;
}
```

**編譯：**

```bash
gcc demo.c -o demo \
    $(pkg-config --cflags --libs libavformat libavcodec libavutil)
```

---

## 相關筆記
- [[FFmpeg模組種類總覽]] — 七個模組概覽
- [[demuxer-module]] — 通用 demuxer 模組實作（FFmpeg API）
- [[libavcodec]] — 解碼 API（avcodec_send_packet / avcodec_receive_frame）
- [[AVIOContext]] — 自訂 I/O 實作（avio_alloc_context）
- [[low_latency_pipeline]] — 低延遲串流架構
- [[V4L2]] — Linux 攝影機裝置介面